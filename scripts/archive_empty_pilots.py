"""Preserve failed zero-record pilots before a protocol upgrade (standard library only)."""

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path


@contextmanager
def run_lock(directory):
    path = directory / ".writer.lock"
    if path.is_symlink():
        raise RuntimeError("Symbolic writer lock is not supported")
    stream = path.open("a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError("A pilot writer is still active; stop it before upgrading") from exc
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def archive_empty_pilots(suite, previous_revision):
    suite = Path(suite).resolve()
    candidates = []
    archive_base = suite / "failed-empty-pilots"
    if archive_base.is_symlink():
        raise RuntimeError("Archive directory must remain inside the suite")
    with ExitStack() as locks:
        for name in ("pilot-one", "pilot-five", "pilot-twenty"):
            run = suite / name
            if not (run / "manifest.json").exists():
                continue
            if run.is_symlink() or run.resolve().parent != suite:
                raise RuntimeError("Pilot directory must remain inside the suite")
            locks.enter_context(run_lock(run))
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise RuntimeError("Invalid pilot manifest; preserve it for manual diagnosis")
            journal = run / "journal.jsonl"
            if journal.is_symlink() or (journal.exists() and journal.read_bytes()):
                raise RuntimeError(
                    "Code changed after annotation began: a nonempty journal exists. Use a new EASCCA_OUTPUTS directory."
                )
            state = run / "state.sqlite"
            if state.is_symlink():
                raise RuntimeError("Symbolic pilot state is not supported")
            if state.exists():
                conn = sqlite3.connect(state.resolve().as_uri() + "?mode=ro", uri=True)
                try:
                    if conn.execute("SELECT count(*) FROM records").fetchone()[0]:
                        raise RuntimeError(
                            "Code changed after annotation began: accepted records exist. Use a new EASCCA_OUTPUTS directory."
                        )
                finally:
                    conn.close()
            candidates.append(run)
        if not candidates:
            return None
        # Validate every candidate before moving any; accepted data is never archived here.
        archive_base.mkdir(exist_ok=True)
        prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", previous_revision[:16]) + "-"
        destination = Path(tempfile.mkdtemp(prefix=prefix, dir=archive_base)).resolve()
        if suite not in destination.parents:
            raise RuntimeError("Archive path escapes the suite")
        metadata = {
            "reason": "protocol_upgrade_after_zero_accepted_annotations",
            "previous_code_revision": previous_revision,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "accepted_records": 0,
            "archived_runs": [run.name for run in candidates],
        }
        (destination / "archive.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        # Windows cannot rename directories containing an open lock file. Production
        # runs on Spark/Linux, where the per-run locks remain held through each rename.
        if os.name == "nt":
            locks.close()
        for run in candidates:
            target = destination / run.name
            if run.resolve().parent != suite or suite not in target.resolve().parents:
                raise RuntimeError("Archive move must stay inside the suite")
            run.rename(target)
        return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--previous-revision", required=True)
    args = parser.parse_args()
    try:
        archived = archive_empty_pilots(args.suite, args.previous_revision)
    except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError) as exc:
        print(f"Cannot upgrade this suite: {exc}", file=sys.stderr)
        return 2
    if archived:
        print(f"Preserved failed zero-record pilots and their audits: {archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
