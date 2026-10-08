"""Frozen engineering pilot selections; these proxies do not establish cue diversity."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .util import atomic_json, digest, outside, within

PLAN_VERSION = 1
STAGES = ("one", "five", "twenty")


def load_plan(path, index, config):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    fingerprint = plan.pop("plan_fingerprint", None)
    if fingerprint != digest(plan):
        raise ValueError("Pilot selection fingerprint mismatch")
    plan["plan_fingerprint"] = fingerprint
    if (
        plan.get("version") != PLAN_VERSION
        or plan.get("dataset_fingerprint") != index.manifest["dataset_fingerprint"]
        or plan.get("config_fingerprint") != config.fingerprint
    ):
        raise ValueError("Pilot selection belongs to a different dataset/configuration/version")
    return plan


def selected_units(path, stage, index, config):
    plan = load_plan(path, index, config)
    if stage not in STAGES:
        raise ValueError("Select pilot stage one, five or twenty")
    available = {u.key: u for u in index.units()}
    selected = []
    seen = set()
    for value in plan["stages"][stage]["units"]:
        key = f"session{value['session']}/window{value['window']}/track_{value['track']}"
        if key in seen or key not in available or asdict(available[key]) != value:
            raise ValueError("Invalid or duplicate unit in pilot selection")
        selected.append(available[key])
        seen.add(key)
    window_count = len({(u.session, u.window) for u in selected})
    if not selected or window_count > 20:
        raise ValueError("Invalid pilot scope")
    if stage == "one" and len(selected) != 1:
        raise ValueError("Stage one requires one track-window")
    if stage == "five" and (len(selected), window_count) != (5, 5):
        raise ValueError("Stage five requires five tracks in five distinct windows")
    if stage == "twenty" and window_count != 20:
        raise ValueError("Stage twenty requires twenty distinct classroom windows")
    return selected


def _spread(candidates, count, initial=(), distinct_windows=False):
    selected = list(initial)
    while len(selected) < count:
        remaining = [
            p
            for p in candidates
            if p not in selected
            and (
                not distinct_windows
                or (p["unit"].session, p["unit"].window)
                not in {(s["unit"].session, s["unit"].window) for s in selected}
            )
        ]
        if not remaining:
            raise ValueError("Insufficient eligible candidates for the requested pilot")

        def score(p):
            if not selected:
                return p["coverage"], p["area"], -p["unit"].track
            unseen_session = p["unit"].session not in {s["unit"].session for s in selected}
            distance = min(
                abs(p["coverage"] - s["coverage"])
                + abs(p["area"] - s["area"])
                + abs(p["position"] - s["position"])
                for s in selected
            )
            return unseen_session, distance, p["coverage"]

        selected.append(max(remaining, key=score))
    return selected


def plan_pilots(index, config, destination, tracks_per_window=3, progress=None):
    destination = outside(destination, index.root)
    if not 1 <= tracks_per_window <= 10:
        raise ValueError("tracks_per_window must be between 1 and 10")
    index.verify_metadata(config)
    if destination.exists():
        plan = load_plan(destination, index, config)
        if plan["tracks_per_window"] != tracks_per_window:
            raise ValueError("Existing pilot selection has a different track count")
        for stage in STAGES:
            selected_units(destination, stage, index, config)
        return plan
    progress = progress or (lambda s: None)
    counts = {
        tuple(row[:3]): row[3]
        for row in index.conn.execute(
            "SELECT session,window,track,count(*) FROM files GROUP BY session,window,track"
        )
    }
    units = index.units()
    max_window = defaultdict(int)
    for unit in units:
        max_window[unit.session] = max(max_window[unit.session], unit.window)
    global_areas = {}
    profiles = []
    exclusions = []
    for unit in units:
        window = (unit.session, unit.window)
        crops = counts.get((*window, unit.track), 0)
        if not crops or counts.get((*window, -1), 0) != config.window_frames:
            continue
        try:
            if window not in global_areas:
                row = index.conn.execute(
                    "SELECT path FROM files WHERE session=? AND window=? AND track=-1 ORDER BY frame LIMIT 1",
                    window,
                ).fetchone()
                with Image.open(within(index.root, row[0])) as image:
                    global_areas[window] = image.width * image.height
            rows = index.conn.execute(
                "SELECT path FROM files WHERE session=? AND window=? AND track=? ORDER BY frame",
                (*window, unit.track),
            ).fetchall()
            areas = []
            for pos in sorted({0, len(rows) // 2, len(rows) - 1}):
                with Image.open(within(index.root, rows[pos][0])) as image:
                    areas.append(image.width * image.height / global_areas[window])
            profiles.append(
                {
                    "unit": unit,
                    "coverage": crops / config.window_frames,
                    "area": min(1.0, statistics.median(areas)),
                    "position": unit.window / max_window[unit.session],
                }
            )
        except (OSError, ValueError) as exc:
            exclusions.append({"unit": unit.key, "reason": str(exc)})
        if len(profiles) and len(profiles) % 250 == 0:
            progress(f"Measured crop size/coverage for {len(profiles)} eligible tracks")
    window_count = len({(p["unit"].session, p["unit"].window) for p in profiles})
    if window_count < 20:
        raise ValueError(
            f"All pilots require 20 windows with 250 global frames and readable crops; found {window_count}. "
            "No reduced pilot will be silently called complete."
        )
    median_area = statistics.median(p["area"] for p in profiles)
    one = max(profiles, key=lambda p: (p["coverage"], -abs(p["area"] - median_area)))
    five = _spread(profiles, 5, [one], distinct_windows=True)
    windows = _spread(profiles, 20, five, distinct_windows=True)
    grouped = defaultdict(list)
    for p in profiles:
        grouped[(p["unit"].session, p["unit"].window)].append(p)
    twenty = []
    for representative in windows:
        unit = representative["unit"]
        candidates = grouped[(unit.session, unit.window)]
        twenty.extend(_spread(candidates, min(tracks_per_window, len(candidates)), [representative]))
    plan = {
        "version": PLAN_VERSION,
        "dataset_fingerprint": index.manifest["dataset_fingerprint"],
        "config_fingerprint": config.fingerprint,
        "tracks_per_window": tracks_per_window,
        "policy": "deterministic spread across sessions, window positions, crop coverage and native crop area",
        "quality_reviewed": False,
        "limitations": "Automatic engineering selection does not establish occlusion, writing, reading or object-activity diversity. Human review is required.",
        "excluded_unreadable_tracks": exclusions,
        "stages": {},
    }
    for name, values in zip(STAGES, ([one], five, twenty)):
        ordered = sorted(values, key=lambda p: p["unit"])
        plan["stages"][name] = {
            "units": [asdict(p["unit"]) for p in ordered],
            "selection_proxies": [
                {"unit": p["unit"].key, **{k: v for k, v in p.items() if k != "unit"}} for p in ordered
            ],
            "classroom_windows": len({(p["unit"].session, p["unit"].window) for p in ordered}),
            "expected_frames": len(ordered) * config.window_frames,
        }
    plan["plan_fingerprint"] = digest(plan)
    atomic_json(destination, plan)
    return plan
