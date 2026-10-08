from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .util import canonical, file_sha256, outside, within

INDEX_VERSION = 1
FRAME_RE = re.compile(r"frame_(\d+)\.(?:jpg|jpeg)", re.IGNORECASE)


@dataclass(frozen=True, order=True)
class Unit:
    session: int
    window: int
    track: int
    start: int

    @property
    def key(self) -> str:
        return f"session{self.session}/window{self.window}/track_{self.track}"


def numeric_dirs(root: Path, prefix: str) -> list[tuple[int, Path]]:
    expression = re.compile(re.escape(prefix) + r"(\d+)")
    result = []
    ids = set()
    for path in root.iterdir():
        match = expression.fullmatch(path.name)
        if path.is_dir() and match:
            number = int(match[1])
            if path.is_symlink() or number in ids:
                raise ValueError(f"Duplicate ID or symbolic directory: {path}")
            ids.add(number)
            result.append((number, path))
    return sorted(result)


def image_files(root: Path) -> dict[int, Path]:
    result = {}
    if not root.is_dir():
        return result
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = FRAME_RE.fullmatch(path.name)
        if match:
            frame = int(match[1])
            if path.is_symlink() or frame in result:
                raise ValueError(f"Duplicate frame or symbolic source: {path}")
            result[frame] = path
        elif path.suffix.lower() in {".jpg", ".jpeg"}:
            raise ValueError(f"Unrecognised frame filename: {path}")
    return dict(sorted(result.items()))


def walk_windows(root: Path, config: Config, sessions=None, windows=None, excluded_windows=None):
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")
    for session, session_path in numeric_dirs(root, "session"):
        if sessions and session not in sessions:
            continue
        if str(session) not in config.session_start_frames:
            raise ValueError(f"Set session_start_frames for session{session}; do not guess the first frame")
        for window, path in numeric_dirs(session_path, "window"):
            if windows and window not in windows:
                continue
            if (session, window) in {tuple(pair) for pair in (excluded_windows or [])}:
                continue
            if window < 1:
                raise ValueError(f"Window IDs begin at 1: {path}")
            start = config.session_start_frames[str(session)] + (window - 1) * config.window_frames
            yield session, window, start, path


def preflight(root: Path, config: Config, sessions=None, windows=None, excluded_windows=None) -> dict:
    root = root.expanduser().resolve()
    excluded_windows = sorted({tuple(pair) for pair in (excluded_windows or [])})
    if any(len(pair) != 2 or any(type(n) is not int or n < 1 for n in pair) for pair in excluded_windows):
        raise ValueError("Excluded windows require positive integer (session, window) pairs")
    result = {
        "root": str(root),
        "fps": config.fps,
        "window_frames": config.window_frames,
        "windows": 0,
        "track_windows": 0,
        "global_files": 0,
        "crop_files": 0,
        "expected_track_frames": 0,
        "missing_crop_positions": 0,
        "issues": [],
        "sessions": {},
        "excluded_windows": [{"session": s, "window": w} for s, w in excluded_windows],
    }
    metadata = hashlib.sha256()
    for session, window, start, path in walk_windows(root, config, sessions, windows, excluded_windows):
        expected = set(range(start, start + config.window_frames))
        global_files = image_files(path / "global_frames")
        crop_root = path / "cropped_frames_per_student"
        partial = path / "cropped_frames_per_student.__partial"
        problems = []
        if partial.exists() or not crop_root.is_dir():
            problems.append("crop_root_not_finalised")
        missing_global = sorted(expected - set(global_files))
        unexpected_global = sorted(set(global_files) - expected)
        if missing_global:
            problems.append(f"missing_global_frames:{len(missing_global)}")
        if unexpected_global:
            problems.append(f"out_of_range_global_frames:{len(unexpected_global)}")
        tracks = numeric_dirs(crop_root, "track_") if crop_root.is_dir() else []
        if not tracks:
            problems.append("no_tracks")
        metadata.update(canonical([session, window, start, [t for t, _ in tracks]]).encode())
        images = [(-1, global_files)]
        for track, track_path in tracks:
            crops = image_files(track_path)
            if set(crops) - expected:
                problems.append(f"out_of_range_crop_frames:track_{track}")
            images.append((track, crops))
            result["crop_files"] += len(crops)
            result["missing_crop_positions"] += len(expected - set(crops))
        for track, files in images:
            for frame, source in files.items():
                stat = source.stat()
                metadata.update(
                    canonical([source.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns]).encode()
                )
        result["windows"] += 1
        result["track_windows"] += len(tracks)
        result["global_files"] += len(global_files)
        result["expected_track_frames"] += len(tracks) * config.window_frames
        result["sessions"].setdefault(str(session), []).append(window)
        if problems:
            result["issues"].append(
                {
                    "session": session,
                    "window": window,
                    "issues": problems,
                    "missing_global_frames": missing_global,
                    "unexpected_global_frames": unexpected_global,
                }
            )
    if not result["windows"]:
        raise ValueError("No matching windows found")
    result["metadata_fingerprint"] = metadata.hexdigest()
    result["missing_windows_against_historical_scope"] = {
        s: sorted(set(range(1, n + 1)) - set(result["sessions"].get(s, [])))
        for s, n in config.expected_session_windows.items()
        if set(range(1, n + 1)) - set(result["sessions"].get(s, []))
    }
    result["historical_full_inventory"] = all(
        result["sessions"].get(s, []) == list(range(1, n + 1))
        for s, n in config.expected_session_windows.items()
    ) and set(result["sessions"]) == set(config.expected_session_windows)
    result["scope"] = (
        "historical_full_inventory" if result["historical_full_inventory"] else "subset_or_changed_inventory"
    )
    return result


def build_index(
    root: Path,
    destination: Path,
    config: Config,
    sessions=None,
    windows=None,
    progress=None,
    excluded_windows=None,
) -> dict:
    root = root.expanduser().resolve()
    destination = outside(destination, root)
    if destination.exists():
        raise FileExistsError(f"Index exists; choose a new index path: {destination}")
    summary = preflight(root, config, sessions, windows, excluded_windows)
    fatal = [
        i
        for i in summary["issues"]
        if any(p == "crop_root_not_finalised" or p.startswith("out_of_range") for p in i["issues"])
    ]
    if fatal:
        raise ValueError(f"Index blocked by unfinished/out-of-range inputs: {fatal[:5]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".building-{uuid.uuid4().hex}")
    conn = sqlite3.connect(temporary)
    fingerprint = hashlib.sha256()
    try:
        conn.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE windows (session INTEGER, window INTEGER, start INTEGER,
              PRIMARY KEY(session,window));
            CREATE TABLE tracks (session INTEGER, window INTEGER, track INTEGER,
              PRIMARY KEY(session,window,track));
            CREATE TABLE files (session INTEGER, window INTEGER, track INTEGER, frame INTEGER,
              path TEXT UNIQUE NOT NULL, sha256 TEXT NOT NULL, size INTEGER, mtime_ns INTEGER,
              PRIMARY KEY(session,window,track,frame));
        """)
        for session, window, start, path in walk_windows(root, config, sessions, windows, excluded_windows):
            conn.execute("INSERT INTO windows VALUES (?,?,?)", (session, window, start))
            tracks = numeric_dirs(path / "cropped_frames_per_student", "track_")
            fingerprint.update(canonical([session, window, start, [t for t, _ in tracks]]).encode())
            images = [(-1, image_files(path / "global_frames"))]
            for track, track_path in tracks:
                conn.execute("INSERT INTO tracks VALUES (?,?,?)", (session, window, track))
                images.append((track, image_files(track_path)))
            for track, files in images:
                for frame, source in files.items():
                    relative = source.relative_to(root).as_posix()
                    before = source.stat()
                    sha = file_sha256(source)
                    after = source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError(f"Source changed during indexing: {source}")
                    conn.execute(
                        "INSERT INTO files VALUES (?,?,?,?,?,?,?,?)",
                        (session, window, track, frame, relative, sha, after.st_size, after.st_mtime_ns),
                    )
                    fingerprint.update(canonical([relative, sha]).encode())
            conn.commit()
            if progress:
                progress(f"Indexed session{session}/window{window} ({len(tracks)} tracks)")
        summary["dataset_fingerprint"] = fingerprint.hexdigest()
        summary["index_version"] = INDEX_VERSION
        summary["session_start_frames"] = config.session_start_frames
        summary["selection"] = {"sessions": sorted(sessions or []), "windows": sorted(windows or [])}
        if excluded_windows:
            summary["selection"]["excluded_windows"] = [
                list(pair) for pair in sorted({tuple(p) for p in excluded_windows})
            ]
        conn.execute("INSERT INTO meta VALUES ('manifest',?)", (canonical(summary),))
        conn.commit()
        conn.close()
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        conn.close()
        temporary.unlink(missing_ok=True)
        raise
    return summary


class DatasetIndex:
    def __init__(self, path: Path, root: Path):
        self.root = root.expanduser().resolve()
        self.path = path.expanduser().resolve()
        self.conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        row = self.conn.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()
        if not row:
            raise ValueError("Missing index manifest")
        self.manifest = json.loads(row[0])
        if self.manifest["index_version"] != INDEX_VERSION:
            raise ValueError("Unsupported index version")

    def close(self):
        self.conn.close()

    def units(self, sessions=None, windows=None, tracks=None) -> list[Unit]:
        rows = self.conn.execute(
            "SELECT t.session,t.window,t.track,w.start FROM tracks t JOIN windows w "
            "ON t.session=w.session AND t.window=w.window ORDER BY t.session,t.window,t.track"
        )
        return [
            Unit(*row)
            for row in rows
            if (not sessions or row[0] in sessions)
            and (not windows or row[1] in windows)
            and (not tracks or row[2] in tracks)
        ]

    def frames(self, unit: Unit, frames) -> list[dict]:
        ids = list(frames)
        if not ids:
            return []
        data = self.conn.execute(
            "SELECT * FROM files WHERE session=? AND window=? AND track IN (-1,?) AND frame BETWEEN ? AND ?",
            (unit.session, unit.window, unit.track, min(ids), max(ids)),
        )
        lookup = {(row["track"], row["frame"]): dict(row) for row in data}
        result = []
        for frame in ids:
            global_source = lookup.get((-1, frame))
            crop_source = lookup.get((unit.track, frame))
            result.append(
                {
                    "session": unit.session,
                    "window": unit.window,
                    "track": unit.track,
                    "source_frame": frame,
                    "time_seconds": frame / 25,
                    "global_source": global_source,
                    "crop_source": crop_source,
                    "input_status": "matched"
                    if global_source and crop_source
                    else "crop_missing"
                    if global_source
                    else "global_missing"
                    if crop_source
                    else "both_missing",
                }
            )
        return result

    def verify_metadata(self, config: Config) -> None:
        if self.manifest["session_start_frames"] != config.session_start_frames:
            raise ValueError("Index and configuration disagree about source frame starts")
        selection = self.manifest["selection"]
        current = preflight(
            self.root, config, selection["sessions"], selection["windows"], selection.get("excluded_windows")
        )
        if current["metadata_fingerprint"] != self.manifest["metadata_fingerprint"]:
            raise ValueError("Dataset inventory changed; build a new index and start a new run")

    def verify_sources(self) -> int:
        count = 0
        for row in self.conn.execute("SELECT path,sha256 FROM files"):
            path = within(self.root, row[0])
            if not path.is_file() or file_sha256(path) != row[1]:
                raise ValueError(f"Indexed source changed or missing: {path}")
            count += 1
        return count
