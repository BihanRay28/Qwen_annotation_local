from __future__ import annotations

import html
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from .localisation import highlight
from .ontology import CUES
from .persistence import event_records, inspect_run, journal_events
from .runner import ImageCache
from .util import atomic_json, canonical, outside


def summarise(records: list[dict]) -> dict:
    records = sorted(records, key=lambda r: r["source_frame"])
    first = records[0]
    counts = {c: Counter(r["cues"][c] for r in records) for c in CUES}
    intervals = {}
    for cue in CUES:
        spans = []
        start = previous = None
        for record in records:
            frame = record["source_frame"]
            supported = record["cues"][cue] == "present"
            if start is not None and (not supported or frame != previous + 1):
                spans.append(
                    {
                        "start_frame": start,
                        "end_frame_exclusive": previous + 1,
                        "duration_seconds": (previous + 1 - start) / 25,
                    }
                )
                start = previous = None
            if supported:
                if start is None:
                    start = frame
                previous = frame
        if start is not None:
            spans.append(
                {
                    "start_frame": start,
                    "end_frame_exclusive": previous + 1,
                    "duration_seconds": (previous + 1 - start) / 25,
                }
            )
        intervals[cue] = spans
    return {
        "session": first["session"],
        "window": first["window"],
        "track": first["track"],
        "committed_positions": len(records),
        "cue_counts": counts,
        "supported_intervals": intervals,
        "input_status_counts": Counter(r["input_status"] for r in records),
        "review_flags": sorted({flag for r in records for flag in r["review_flags"]}),
        "track_eligibility": "unreviewed",
    }


def export_run(
    run_dir: Path, destination: Path, dataset_root: Path | None = None, review_images: int = 0
) -> dict:
    status = inspect_run(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    source_root = dataset_root or Path(manifest["dataset_root"])
    destination = outside(destination, source_root)
    if destination == run_dir.resolve() or run_dir.resolve() in destination.parents:
        raise ValueError("Choose a separate export directory outside the run directory")
    if destination.exists():
        raise FileExistsError("Export directory exists; choose a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eascca-export-", dir=destination.parent) as temporary:
        working = Path(temporary)
        conn = sqlite3.connect(working / "sort.sqlite")
        conn.execute("CREATE TABLE records (session INT, window INT, track INT, frame INT, value TEXT)")
        for event in journal_events(run_dir / "journal.jsonl"):
            conn.executemany(
                "INSERT INTO records VALUES (?,?,?,?,?)",
                [
                    (r["session"], r["window"], r["track"], r["source_frame"], canonical(r))
                    for r in event_records(event)
                ],
            )
        conn.commit()
        output = working / "export"
        output.mkdir()
        reviewed = []
        with (
            (output / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as frames,
            (output / "track_windows.jsonl").open("w", encoding="utf-8", newline="\n") as summaries,
        ):
            group = []
            group_key = None
            rows = conn.execute(
                "SELECT session,window,track,frame,value FROM records ORDER BY session,window,track,frame"
            )
            for session, window, track, frame, value in rows:
                record = json.loads(value)
                current = (session, window, track)
                if group and current != group_key:
                    summaries.write(canonical(summarise(group)) + "\n")
                    group = []
                group_key = current
                group.append(record)
                frames.write(value + "\n")
                if len(reviewed) < review_images:
                    reviewed.append(record)
            if group:
                summaries.write(canonical(summarise(group)) + "\n")
            for stream in (frames, summaries):
                stream.flush()
                os.fsync(stream.fileno())
        conn.close()
        if review_images:
            write_review(output, source_root, reviewed)
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "validation.json", status)
        os.replace(output, destination)
    return {**status, "export_directory": str(destination), "review_image_records": len(reviewed)}


def write_review(output: Path, source_root: Path, records: list[dict]):
    cache = ImageCache(source_root.expanduser().resolve(), 40)
    image_dir = output / "review_images"
    image_dir.mkdir()
    cards = []
    for i, record in enumerate(records):
        global_image = (
            cache.load({"path": record["global_path"], "sha256": record["global_sha256"]}, cached=True)
            if record["global_path"]
            else None
        )
        crop = (
            cache.load({"path": record["crop_path"], "sha256": record["crop_sha256"]})
            if record["crop_path"]
            else None
        )
        canvas = Image.new("RGB", (1200, 650), "white")
        if global_image is not None:
            global_image = highlight(global_image, record["localisation"])
            fitted = ImageOps.contain(global_image, (960, 650))
            canvas.paste(fitted, (0, 0))
        if crop is not None:
            fitted = ImageOps.contain(crop, (230, 650))
            canvas.paste(fitted, (970, 0))
        name = f"review_images/{i:04d}.png"
        canvas.save(output / name)
        title = f"session{record['session']} window{record['window']} track_{record['track']} frame {record['source_frame']}"
        cards.append(
            f"<section><h2>{html.escape(title)}</h2><img src='{name}' alt='Matched source evidence'>"
            f"<pre>{html.escape(json.dumps(record, indent=2))}</pre></section>"
        )
    (output / "review.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Local cue annotation review</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem}img{max-width:100%}"
        "pre{white-space:pre-wrap}section{margin-bottom:3rem}</style><h1>Local cue annotation review</h1>"
        "<p>Machine proposals require human review. This page includes a bounded prefix of the export; "
        "it is not a representative evaluation sample. No engagement scores are assigned.</p>"
        + "".join(cards),
        encoding="utf-8",
    )
