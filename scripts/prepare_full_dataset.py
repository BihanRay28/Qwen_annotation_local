"""Retire only known annotation outputs; never remove or move source dataset directories."""

import argparse
import json
import os
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

from archive_empty_pilots import run_lock

FORMAT_PREFIX = b'{"format":"eascca-indexed-labels-1.0"'


def owned_labels(path):
    if path.is_symlink():
        raise ValueError("Labels file must not be a symbolic link")
    with path.open("rb") as stream:
        return stream.read(len(FORMAT_PREFIX)) == FORMAT_PREFIX


def prepare(outputs, dataset, repo, revision, config, reset=False, filename="labels.json"):
    outputs, dataset, repo = (Path(p).resolve() for p in (outputs, dataset, repo))
    for source in (dataset, repo):
        if outputs == source or outputs in source.parents or source in outputs.parents:
            raise ValueError("Working outputs must be separate from the dataset and repository")
    if filename not in {"labels.json", "labels.py"}:
        raise ValueError("The root output must be labels.json or labels.py")
    outputs.mkdir(parents=True, exist_ok=True)
    suite = outputs / "full-dataset"
    if suite.is_symlink():
        raise ValueError("Full run directory cannot be symbolic")
    marker = suite / "setup.json"
    desired = {
        "revision": revision,
        "config": json.loads(Path(config).read_text(encoding="utf-8")),
        "dataset": str(dataset),
        "filename": filename,
    }
    if marker.exists() and not reset:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if any(saved.get(k) != v for k, v in desired.items()):
            raise ValueError("Run setup changed. Use EASCCA_RESET=1 to archive generated results and restart")
        return saved
    labels = dataset / filename
    if labels.exists() and not owned_labels(labels):
        raise ValueError(
            f"Preserving existing unrecognised dataset file: {labels}; choose the other filename"
        )
    archive_base = outputs / "reset-archives"
    if archive_base.is_symlink():
        raise ValueError("Archive directory cannot be symbolic")
    candidates = [p for p in (outputs / "automatic-pilots", suite) if p.exists()]
    with ExitStack() as locks:
        for candidate in candidates:
            if candidate.is_symlink() or candidate.resolve().parent != outputs:
                raise ValueError("Generated run directory escapes the outputs root")
            locks.enter_context(run_lock(candidate))
            # Pilot runners lock .suite.lock; full runs also hold a top-level command lock.
            for nested in candidate.glob("pilot-*"):
                if nested.is_dir():
                    if nested.is_symlink() or nested.resolve().parent != candidate:
                        raise ValueError("Pilot run escapes its generated suite")
                    locks.enter_context(run_lock(nested))
            full_run = candidate / "run"
            if full_run.is_dir():
                if full_run.is_symlink() or full_run.resolve().parent != candidate:
                    raise ValueError("Full run escapes its generated suite")
                locks.enter_context(run_lock(full_run))
            lock_path = candidate / ".suite.lock"
            if lock_path.is_symlink():
                raise ValueError("Symbolic suite lock is unsupported")
            if os.name != "nt":
                import fcntl

                stream = locks.enter_context(lock_path.open("a+b"))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive_base.mkdir(exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="reset-", dir=archive_base)).resolve()
        if outputs not in destination.parents:
            raise ValueError("Archive must remain in generated outputs")
        if os.name == "nt":
            locks.close()
        for candidate in candidates:
            target = destination / candidate.name
            if candidate.resolve().parent != outputs or outputs not in target.resolve().parents:
                raise ValueError("Refusing to move a source directory")
            candidate.rename(target)
        if labels.exists():
            # Only the explicitly recognised generated labels file may leave the dataset.
            if labels.resolve().parent != dataset or not owned_labels(labels):
                raise ValueError("Refusing to move an original dataset file")
            shutil.copyfile(labels, destination / filename)
            # The live file is replaced atomically after new inference has started.
        reusable = next(
            (
                p
                for p in (
                    destination / "full-dataset/index.sqlite",
                    destination / "automatic-pilots/index.sqlite",
                )
                if p.is_file() and not p.is_symlink()
            ),
            None,
        )
        suite.mkdir()
        if reusable:
            shutil.copyfile(reusable, suite / "index.sqlite")
        shutil.copyfile(config, suite / "config.json")
        desired["archive"] = str(destination)
        marker.write_text(json.dumps(desired, indent=2) + "\n", encoding="utf-8")
        (suite / "labels.json").write_text(
            '{"format":"eascca-indexed-labels-1.0","metadata":{"complete":false,'
            '"status":"setup","quality_reviewed":false},"sessions":{}}\n',
            encoding="utf-8",
        )
        (destination / "reset.json").write_text(json.dumps(desired, indent=2) + "\n", encoding="utf-8")
        return desired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("outputs", "dataset", "repo", "config"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--filename", default="labels.json")
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()
    print(
        json.dumps(
            prepare(
                args.outputs, args.dataset, args.repo, args.revision, args.config, args.reset, args.filename
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
