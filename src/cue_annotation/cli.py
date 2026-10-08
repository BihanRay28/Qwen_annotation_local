from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .util import atomic_json, outside


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def compact_summary(value):
    summary = dict(value)
    if "sessions" in summary:
        summary["sessions"] = {
            s: {"window_count": len(w), "first_window": min(w), "last_window": max(w)}
            for s, w in summary["sessions"].items()
            if w
        }
    if "missing_windows_against_historical_scope" in summary:
        summary["missing_window_counts_against_historical_scope"] = {
            s: len(w) for s, w in summary.pop("missing_windows_against_historical_scope").items()
        }
    if summary.get("first_chunk"):
        summary["first_chunk"] = {k: v for k, v in summary["first_chunk"].items() if k != "sources"}
    return summary


def parser():
    result = argparse.ArgumentParser(description="Local observable cue annotation with Qwen2.5-VL")
    commands = result.add_subparsers(dest="command", required=True)

    def data_options(command):
        command.add_argument("--dataset-root", type=Path, default=Path("~/Desktop/preprocessed_dataset"))
        command.add_argument("--config", type=Path)

    def filters(command):
        command.add_argument("--session", type=int, nargs="+", dest="sessions")
        command.add_argument("--window", type=int, nargs="+", dest="windows")

    preflight = commands.add_parser("preflight", help="Read-only inventory and contract reconciliation")
    data_options(preflight)
    filters(preflight)
    preflight.add_argument("--output", type=Path)
    preflight.add_argument("--require-full-inventory", action="store_true")

    index = commands.add_parser("index", help="Build a content-hashed source index outside the dataset")
    data_options(index)
    filters(index)
    index.add_argument("--output", required=True, type=Path)

    download = commands.add_parser("download-model", help="Explicit online download of open model weights")
    download.add_argument("--config", type=Path)
    download.add_argument("--cache-dir", type=Path)

    run = commands.add_parser("run", help="Annotate or resume; --dry-run produces no labels")
    data_options(run)
    filters(run)
    run.add_argument("--index", required=True, type=Path)
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--model-dir", type=Path)
    run.add_argument("--track", type=int, nargs="+", dest="tracks")
    run.add_argument("--max-track-windows", type=int)
    run.add_argument("--max-chunks", type=int, help="Graceful stop after this many new accepted chunks")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--stage", choices=["pilot", "bulk"], default="pilot")
    run.add_argument("--pilot-approval", type=Path)

    units = commands.add_parser("list-units", help="List actual indexed track-window IDs in numeric order")
    data_options(units)
    filters(units)
    units.add_argument("--index", required=True, type=Path)
    units.add_argument("--limit", type=int, default=20)

    status = commands.add_parser("status", help="Read committed journal progress")
    status.add_argument("--run-dir", required=True, type=Path)

    validate = commands.add_parser(
        "validate", help="Validate journal ownership or verify an index's source hashes"
    )
    validate.add_argument("--run-dir", type=Path)
    validate.add_argument("--index", type=Path)
    validate.add_argument("--dataset-root", type=Path, default=Path("~/Desktop/preprocessed_dataset"))
    validate.add_argument("--require-complete", action="store_true")

    export = commands.add_parser(
        "export", help="Export per-frame JSONL, intervals and optional local review images"
    )
    export.add_argument("--run-dir", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--dataset-root", type=Path)
    export.add_argument(
        "--review-images", type=int, default=0, help="Bounded prefix for inspection, not an evaluation sample"
    )

    evaluation = commands.add_parser(
        "evaluate", help="Score predictions against independently adjudicated reference JSONL"
    )
    evaluation.add_argument("--annotations", required=True, type=Path)
    evaluation.add_argument("--reference", required=True, type=Path)
    evaluation.add_argument("--output", required=True, type=Path)
    return result


def verify_approval(path, index, config, identity):
    from .ontology import ONTOLOGY_FINGERPRINT
    from .prompts import PROMPT_FINGERPRINT
    from .schema import SCHEMA_FINGERPRINT

    if path is None:
        raise ValueError("Bulk inference requires --pilot-approval after independent pilot review")
    approval = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_fingerprint": index.manifest["dataset_fingerprint"],
        "model_artifact_fingerprint": identity["artifact_fingerprint"],
        "config_fingerprint": config.fingerprint,
        "ontology_fingerprint": ONTOLOGY_FINGERPRINT,
        "prompt_fingerprint": PROMPT_FINGERPRINT,
        "schema_fingerprint": SCHEMA_FINGERPRINT,
    }
    if any(approval.get(k) != v for k, v in required.items()):
        raise ValueError("Pilot approval does not match this dataset, model or annotation protocol")
    if approval.get("quality_reviewed") is not True or approval.get("resources_feasible") is not True:
        raise ValueError("Pilot quality and resource review are required before bulk")
    if not approval.get("approved_by", "").strip() or not approval.get("review_notes", "").strip():
        raise ValueError("Pilot approval requires a named reviewer and documented quality decision")


def dispatch(args):
    if args.command in {"preflight", "index"}:
        from .dataset import build_index, preflight

        config = load_config(args.config)
        root = args.dataset_root.expanduser().resolve()
        if args.command == "preflight":
            summary = preflight(root, config, args.sessions, args.windows)
            if args.output:
                atomic_json(outside(args.output, root), summary)
            print_json(compact_summary(summary))
            fatal = any(
                p == "crop_root_not_finalised" or p.startswith("out_of_range")
                for i in summary["issues"]
                for p in i["issues"]
            )
            return (
                2
                if fatal or (args.require_full_inventory and not summary["historical_full_inventory"])
                else 0
            )
        summary = build_index(
            root,
            args.output,
            config,
            args.sessions,
            args.windows,
            progress=lambda s: print(s, file=sys.stderr, flush=True),
        )
        print_json(compact_summary(summary))
        return 0
    if args.command == "download-model":
        from .qwen_backend import download_model

        print_json(
            download_model(load_config(args.config), args.cache_dir.expanduser() if args.cache_dir else None)
        )
        return 0
    if args.command in {"run", "list-units"}:
        from .dataset import DatasetIndex

        config = load_config(args.config)
        index = DatasetIndex(args.index, args.dataset_root)
        try:
            units = index.units(args.sessions, args.windows, getattr(args, "tracks", None))
            if not units:
                raise ValueError("No indexed track-windows match the selection")
            if args.command == "list-units":
                if args.limit < 1:
                    raise ValueError("Limit must be positive")
                print_json({"matching_units": len(units), "units": [u.__dict__ for u in units[: args.limit]]})
                return 0
            for name in ("max_track_windows", "max_chunks"):
                if getattr(args, name) is not None and getattr(args, name) < 1:
                    raise ValueError(f"{name} must be positive")
            if args.max_track_windows:
                units = units[: args.max_track_windows]
            from .runner import dry_run, run_annotation

            if args.dry_run:
                print_json(compact_summary(dry_run(index, config, units, args.run_dir)))
                return 0
            if args.stage == "pilot" and len({(u.session, u.window) for u in units}) > 20:
                raise ValueError(
                    "Pilot scope is at most 20 classroom windows; select units or use reviewed --stage bulk"
                )
            if args.model_dir is None:
                try:
                    from huggingface_hub import snapshot_download
                except ImportError as exc:
                    raise RuntimeError("Install [vlm], download weights, then supply --model-dir") from exc
                model_dir = Path(
                    snapshot_download(config.model_id, revision=config.model_revision, local_files_only=True)
                )
            else:
                model_dir = args.model_dir.expanduser().resolve()
            if args.stage == "bulk":
                from .qwen_backend import model_identity

                verify_approval(args.pilot_approval, index, config, model_identity(model_dir, config))
            summary = run_annotation(
                index,
                config,
                units,
                args.run_dir.expanduser().resolve(),
                model_dir,
                args.max_chunks,
                progress=lambda s: print(s, file=sys.stderr, flush=True),
            )
            print_json(summary)
            return 2 if summary.get("failed_chunks_this_invocation", 0) else 0
        finally:
            index.close()
    if args.command in {"status", "validate"}:
        if args.command == "validate" and args.index:
            if args.run_dir:
                raise ValueError("Validate one index or one run at a time")
            from .dataset import DatasetIndex

            index = DatasetIndex(args.index, args.dataset_root)
            try:
                print_json(
                    {
                        "verified_source_files": index.verify_sources(),
                        "dataset_fingerprint": index.manifest["dataset_fingerprint"],
                    }
                )
                return 0
            finally:
                index.close()
        if args.run_dir is None:
            raise ValueError("Provide --run-dir or --index")
        from .persistence import inspect_run

        summary = inspect_run(args.run_dir.expanduser().resolve())
        print_json(summary)
        return 2 if getattr(args, "require_complete", False) and not summary["complete"] else 0
    if args.command == "export":
        if args.review_images < 0:
            raise ValueError("Review image limit must be nonnegative")
        from .export import export_run

        print_json(
            export_run(
                args.run_dir.expanduser().resolve(), args.output, args.dataset_root, args.review_images
            )
        )
        return 0
    if args.command == "evaluate":
        from .evaluation import evaluate

        result = evaluate(args.annotations, args.reference)
        atomic_json(args.output, result)
        print_json(result)
        return 0
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return dispatch(args)
    except KeyboardInterrupt:
        print("Interrupted; resume with the same command and run directory", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
