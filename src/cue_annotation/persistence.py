from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .schema import key, validate_record
from .util import atomic_json, canonical, digest


@contextmanager
def writer_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
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
            raise RuntimeError("Another process owns this run directory") from exc
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


def journal_events(path: Path, repair_tail: bool = False):
    if not path.exists():
        return
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError(f"Interrupted journal tail at byte {offset}; resume under writer lock")
                tail = path.with_name(path.name + f".interrupted-{offset}")
                if tail.exists() and tail.read_bytes() != line:
                    raise ValueError("Conflicting interrupted-tail audit file")
                with tail.open("wb") as saved:
                    saved.write(line)
                    saved.flush()
                    os.fsync(saved.fileno())
                # The tail is uncommitted; preserve it before truncation.
                with path.open("r+b") as repaired:
                    repaired.truncate(offset)
                    repaired.flush()
                    os.fsync(repaired.fileno())
                break
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid committed journal line at byte {offset}") from exc
            yield event


def event_records(event: dict) -> list[dict]:
    if event.get("type") != "chunk_commit" or not isinstance(event.get("records"), list):
        raise ValueError("Unknown journal event")
    records = [validate_record(r).model_dump() for r in event["records"]]
    keys = [key(r) for r in records]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Empty chunk or duplicate record ownership")
    if event.get("keys") != keys or event.get("digest") != digest(records):
        raise ValueError("Committed record digest/ownership mismatch")
    return records


class RunStore:
    """Open only while holding writer_lock; SQLite is always rebuildable."""

    def __init__(self, root: Path, manifest: dict, repair_tail: bool = True):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.journal = root / "journal.jsonl"
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError("Run identity changed; use a new run directory")
        elif self.journal.exists():
            raise ValueError("Journal exists without its manifest")
        else:
            atomic_json(manifest_path, manifest)
        self.manifest = manifest
        self.unit_ranges = {
            (u["session"], u["window"], u["track"]): (
                u["start"],
                u["start"] + manifest["config"]["window_frames"],
            )
            for u in manifest["units"]
        }
        state = root / "state.sqlite"
        self.conn = sqlite3.connect(state)
        try:
            self.conn.execute("PRAGMA quick_check").fetchall()
        except sqlite3.DatabaseError:
            self.conn.close()
            os.replace(state, root / f"state.corrupt-{time.time_ns()}.sqlite")
            self.conn = sqlite3.connect(state)
        self.conn.executescript("""
            DROP TABLE IF EXISTS records;
            DROP TABLE IF EXISTS chunks;
            CREATE TABLE records (key TEXT PRIMARY KEY, digest TEXT NOT NULL, chunk TEXT NOT NULL);
            CREATE TABLE chunks (id TEXT PRIMARY KEY, digest TEXT NOT NULL, n INTEGER NOT NULL);
        """)
        try:
            for event in journal_events(self.journal, repair_tail=repair_tail):
                self._insert(event)
            self.conn.commit()
        except BaseException:
            self.conn.close()
            raise

    def close(self):
        self.conn.close()

    def _insert(self, event):
        if event.get("run_fingerprint") != self.manifest["run_fingerprint"]:
            raise ValueError("Journal contains a different run identity")
        records = event_records(event)
        for record in records:
            bounds = self.unit_ranges.get((record["session"], record["window"], record["track"]))
            if bounds is None or not bounds[0] <= record["source_frame"] < bounds[1]:
                raise ValueError("Record is outside the declared run scope")
        prior = self.conn.execute("SELECT digest FROM chunks WHERE id=?", (event["chunk_id"],)).fetchone()
        if prior:
            raise ValueError("Duplicate chunk ownership in journal")
        for record in records:
            record_key = key(record)
            prior = self.conn.execute("SELECT digest FROM records WHERE key=?", (record_key,)).fetchone()
            if prior:
                raise ValueError(f"Primary frame belongs to more than one chunk: {record_key}")
            self.conn.execute(
                "INSERT INTO records VALUES (?,?,?)", (record_key, digest(record), event["chunk_id"])
            )
        self.conn.execute(
            "INSERT INTO chunks VALUES (?,?,?)", (event["chunk_id"], event["digest"], len(records))
        )

    def owns(self, keys: list[str]) -> bool:
        count = sum(
            self.conn.execute("SELECT 1 FROM records WHERE key=?", (k,)).fetchone() is not None for k in keys
        )
        if count not in (0, len(keys)):
            return False
        return count == len(keys)

    def contains(self, record_key: str) -> bool:
        return self.conn.execute("SELECT 1 FROM records WHERE key=?", (record_key,)).fetchone() is not None

    def commit(self, chunk_id: str, records: list[dict]) -> None:
        records = [validate_record(r).model_dump() for r in records]
        keys = [key(r) for r in records]
        for record in records:
            bounds = self.unit_ranges.get((record["session"], record["window"], record["track"]))
            if bounds is None or not bounds[0] <= record["source_frame"] < bounds[1]:
                raise ValueError("Record is outside the declared run scope")
        if any(self.contains(k) for k in keys):
            raise ValueError("Commit would overwrite accepted primary records")
        event = {
            "type": "chunk_commit",
            "chunk_id": chunk_id,
            "run_fingerprint": self.manifest["run_fingerprint"],
            "records": records,
            "keys": keys,
            "digest": digest(records),
        }
        event_records(event)
        # One complete newline-terminated event owns a whole validated chunk.
        with self.journal.open("ab") as stream:
            stream.write((canonical(event) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        self._insert(event)
        self.conn.commit()

    def counts(self) -> dict:
        records = self.conn.execute("SELECT count(*) FROM records").fetchone()[0]
        chunks = self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return {
            "committed_frames": records,
            "committed_chunks": chunks,
            "expected_frames": self.manifest["expected_frames"],
            "complete": records == self.manifest["expected_frames"],
        }


def inspect_run(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    seen = set()
    chunks = set()
    for event in journal_events(root / "journal.jsonl"):
        if event.get("run_fingerprint") != manifest["run_fingerprint"]:
            raise ValueError("Journal run identity mismatch")
        records = event_records(event)
        if event["chunk_id"] in chunks or any(key(r) in seen for r in records):
            raise ValueError("Duplicate primary ownership in journal")
        seen.update(key(r) for r in records)
        chunks.add(event["chunk_id"])
    expected = set()
    for unit in manifest["units"]:
        expected.update(
            f"{unit['session']}:{unit['window']}:{unit['track']}:{f}"
            for f in range(unit["start"], unit["start"] + manifest["config"]["window_frames"])
        )
    if seen - expected:
        raise ValueError("Journal owns frames outside the declared run scope")
    return {
        "run_fingerprint": manifest["run_fingerprint"],
        "committed_frames": len(seen),
        "expected_frames": len(expected),
        "remaining_frames": len(expected - seen),
        "committed_chunks": len(chunks),
        "complete": seen == expected,
        "next_missing_key": next(
            (k for k in sorted(expected, key=lambda x: tuple(map(int, x.split(":")))) if k not in seen), None
        ),
    }
