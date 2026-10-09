"""Disk-backed, atomic single-JSON snapshots of a resumable full annotation run."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from .ontology import CUES
from .persistence import event_records, journal_events
from .util import canonical, outside

FORMAT = "eascca-indexed-labels-1.0"


def full_scope(report):
    if any(p.startswith("out_of_range") for i in report["issues"] for p in i["issues"]):
        raise ValueError("Out-of-range source frames cannot be automatically excluded")
    return sorted(
        {(i["session"], i["window"]) for i in report["issues"] if "crop_root_not_finalised" in i["issues"]}
    )


class LabelsSnapshot:
    """Journal is authoritative; the sorting database and published JSON are derived."""

    def __init__(self, run_dir, manifest, inventory, interval=900):
        self.run_dir = Path(run_dir)
        self.output = outside(self.run_dir.parent / "labels.json", Path(manifest["dataset_root"]))
        self.manifest, self.inventory, self.interval = manifest, inventory, interval
        state = self.run_dir / "labels.sqlite"
        self.conn = sqlite3.connect(state)
        try:
            if self.conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise sqlite3.DatabaseError("Corrupt derived label sorting state")
        except sqlite3.DatabaseError:
            self.conn.close()
            os.replace(state, self.run_dir / f"labels.corrupt-{time.time_ns()}.sqlite")
            self.conn = sqlite3.connect(state)
        self.conn.executescript("""
            DROP TABLE IF EXISTS labels;
            CREATE TABLE labels (session INTEGER, window INTEGER, track INTEGER,
                frame INTEGER, value TEXT, PRIMARY KEY(session,window,track,frame));
        """)
        try:
            for event in journal_events(self.run_dir / "journal.jsonl"):
                self.add(event_records(event), publish=False)
            self.conn.commit()
            self.publish()
        except BaseException:
            self.conn.close()
            raise

    def add(self, records, publish=True):
        self.conn.executemany(
            "INSERT INTO labels VALUES (?,?,?,?,?)",
            [(r["session"], r["window"], r["track"], r["source_frame"], canonical(r)) for r in records],
        )
        if publish:
            self.conn.commit()
            if time.monotonic() - self.last_publish >= self.interval:
                self.publish()

    def publish(self):
        counts = {
            (s, w, t): n
            for s, w, t, n in self.conn.execute(
                "SELECT session,window,track,count(*) FROM labels GROUP BY session,window,track"
            )
        }
        total = sum(counts.values())
        inventory = self.inventory
        metadata = {
            "run_fingerprint": self.manifest["run_fingerprint"],
            "dataset_fingerprint": self.manifest["dataset_fingerprint"],
            "model": self.manifest["model"],
            "config": self.manifest["config"],
            "prompt_fingerprint": self.manifest["prompt_fingerprint"],
            "quality_reviewed": False,
            "labels_are": "machine_proposals_for_observable_cues",
            "accepted_frames": total,
            "expected_indexed_frames": self.manifest["expected_frames"],
            "indexed_annotations_complete": total == self.manifest["expected_frames"],
            "all_source_windows_ready": not inventory["issues"],
            "historical_inventory_complete": inventory["historical_full_inventory"],
            "source_issues": inventory["issues"],
            "missing_windows": inventory["missing_windows_against_historical_scope"],
            "index_path": "sessions.sessionN.windows.windowN.tracks.track_N.frames.SOURCE_FRAME",
        }
        metadata["complete"] = (
            metadata["indexed_annotations_complete"]
            and metadata["all_source_windows_ready"]
            and metadata["historical_inventory_complete"]
        )
        units = {}
        for unit in self.manifest["units"]:
            units.setdefault((unit["session"], unit["window"]), []).append(unit)
        issues = {(i["session"], i["window"]): i["issues"] for i in inventory["issues"]}
        sessions = {int(s): set(w) for s, w in inventory["sessions"].items()}
        for session, windows in inventory["missing_windows_against_historical_scope"].items():
            sessions.setdefault(int(session), set()).update(windows)
        temporary = self.output.with_name(self.output.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write('{"format":' + canonical(FORMAT) + ',"metadata":' + canonical(metadata))
                stream.write(',"cue_definitions":' + canonical(CUES) + ',"sessions":{')
                for si, session in enumerate(sorted(sessions)):
                    stream.write(("," if si else "") + canonical(f"session{session}") + ':{"windows":{')
                    for wi, window in enumerate(sorted(sessions[session])):
                        source_issues = issues.get((session, window), [])
                        if window not in inventory["sessions"].get(str(session), []):
                            source_issues = ["source_window_missing"]
                        stream.write(("," if wi else "") + canonical(f"window{window}") + ":{")
                        stream.write('"source_issues":' + canonical(source_issues) + ',"tracks":{')
                        for ti, unit in enumerate(units.get((session, window), [])):
                            track = unit["track"]
                            accepted = counts.get((session, window, track), 0)
                            summary = {
                                "first_source_frame": unit["start"],
                                "expected_frames": self.manifest["config"]["window_frames"],
                                "accepted_frames": accepted,
                                "complete": accepted == self.manifest["config"]["window_frames"],
                            }
                            stream.write(("," if ti else "") + canonical(f"track_{track}") + ":{")
                            stream.write('"coverage":' + canonical(summary) + ',"frames":{')
                            rows = self.conn.execute(
                                "SELECT frame,value FROM labels WHERE session=? AND window=? AND track=? ORDER BY frame",
                                (session, window, track),
                            )
                            for fi, (frame, value) in enumerate(rows):
                                stream.write(("," if fi else "") + json.dumps(str(frame)) + ":" + value)
                            stream.write("}}")
                        stream.write("}}")
                    stream.write("}}")
                stream.write("}}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.output)
        finally:
            temporary.unlink(missing_ok=True)
        self.last_publish = time.monotonic()

    def close(self):
        try:
            self.publish()
        finally:
            self.conn.close()
