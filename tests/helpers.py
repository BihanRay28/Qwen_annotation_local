import json
import re
from pathlib import Path

from PIL import Image

from cue_annotation.config import Config
from cue_annotation.dataset import DatasetIndex, build_index
from cue_annotation.ontology import CUES
from cue_annotation.runner import run_manifest


def image(path: Path, color=(80, 120, 160)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 16), color).save(path)


def dataset(base: Path, count=3, session=5, window=1, track=2):
    root = base / "dataset"
    config = Config()
    start = config.session_start_frames[str(session)] + (window - 1) * 250
    directory = root / f"session{session}/window{window}"
    (directory / f"cropped_frames_per_student/track_{track}").mkdir(parents=True)
    for frame in range(start, start + count):
        image(directory / f"global_frames/frame_{frame:06d}.jpg")
        image(directory / f"cropped_frames_per_student/track_{track}/frame_{frame:06d}.jpg")
    index_path = base / "index.sqlite"
    build_index(root, index_path, config)
    index = DatasetIndex(index_path, root)
    units = index.units()
    manifest = run_manifest(
        index, config, units, {"model_id": "synthetic_test_backend", "artifact_fingerprint": "test-only"}
    )
    return root, config, index, units, manifest


class FakeLocaliser:
    def locate(self, crop, global_image, track_key):
        return (
            {"status": "accepted", "bbox": [0, 0, 4, 4], "score": 1.0, "margin": 1.0}
            if crop and global_image
            else {"status": "unavailable", "reason": "source_missing"}
        )


class FakeBackend:
    """Synthetic schema/recovery fixture. Never exposed by the production CLI."""

    def __init__(self, limit=None, invalid=False):
        self.limit = limit
        self.invalid = invalid
        self.calls = 0
        self.last_metrics = {"synthetic": True}

    def generate(self, messages):
        from cue_annotation.qwen_backend import SplitRequired

        self.calls += 1
        text = next(
            c["text"]
            for c in messages[-1]["content"]
            if c.get("type") == "text" and c["text"].startswith("Annotate ONLY")
        )
        primary = [int(n) for n in re.search(r"once each: ([\d,]+)", text)[1].split(",")]
        if self.limit is not None and len(primary) > self.limit:
            raise SplitRequired("Synthetic token budget fixture")
        if self.invalid:
            return "malformed response"
        return json.dumps({"frames": [model_frame(f) for f in primary]})


def model_frame(frame):
    return {
        "frame": frame,
        "visibility": {
            "body": "visible",
            "head": "visible",
            "face": "visible",
            "eyes": "unknown",
            "hands": "visible",
        },
        "present": [],
        "unknown": list(CUES),
        "confidence": "low",
        "evidence": [],
        "review_flags": [],
    }
