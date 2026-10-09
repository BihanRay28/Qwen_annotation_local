from __future__ import annotations

import hashlib
import io
import os
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from . import __version__
from .chunking import chunks
from .localisation import Localiser, highlight
from .ontology import ONTOLOGY_FINGERPRINT
from .persistence import RunStore, writer_lock
from .prompts import PROMPT_FINGERPRINT, messages
from .qwen_backend import FatalGPUError, QwenBackend, SplitRequired, environment, model_identity
from .schema import (
    SCHEMA_FINGERPRINT,
    PlaceholderEvidenceError,
    expand_response,
    missing_record,
    model_json_text,
)
from .util import atomic_json, canonical, digest, outside, within


class SourceError(RuntimeError):
    pass


class ImageCache:
    def __init__(self, root: Path, maximum: int):
        self.root, self.maximum = root, maximum
        self.cache = OrderedDict()

    def load(self, source: dict | None, cached=False):
        if source is None:
            return None
        cache_key = (source["path"], source["sha256"])
        if cached and cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        path = within(self.root, source["path"])
        try:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise SourceError(f"Source changed after indexing: {path}")
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                rgb = image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise SourceError(f"Unreadable source: {path}: {exc}") from exc
        if cached:
            self.cache[cache_key] = rgb
            while len(self.cache) > self.maximum:
                self.cache.popitem(last=False)
        return rgb


def code_fingerprint() -> str:
    directory = Path(__file__).parent
    return digest(
        [[p.name, hashlib.sha256(p.read_bytes()).hexdigest()] for p in sorted(directory.glob("*.py"))]
    )


def run_manifest(index, config, units, identity) -> dict:
    identity = {k: v for k, v in identity.items() if k != "local_path"}
    manifest = {
        "package_version": __version__,
        "dataset_root": str(index.root),
        "dataset_fingerprint": index.manifest["dataset_fingerprint"],
        "config": config.as_dict(),
        "config_fingerprint": config.fingerprint,
        "model": identity,
        "environment": environment(),
        "code_fingerprint": code_fingerprint(),
        "ontology_fingerprint": ONTOLOGY_FINGERPRINT,
        "prompt_fingerprint": PROMPT_FINGERPRINT,
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "units": [asdict(u) for u in units],
        "expected_frames": len(units) * config.window_frames,
    }
    manifest["run_fingerprint"] = digest(manifest)
    return manifest


def task_iterator(units, config):
    grouped = defaultdict(list)
    for unit in units:
        grouped[(unit.session, unit.window)].append(unit)
    for group in sorted(grouped):
        window_units = grouped[group]
        for chunk in chunks(
            window_units[0].start, config.window_frames, config.primary_frames, config.context_frames
        ):
            for unit in window_units:
                yield unit, chunk


class Runner:
    def __init__(self, index, config, store, backend, localiser, progress=None):
        self.index, self.config, self.store = index, config, store
        self.backend, self.localiser = backend, localiser
        self.images = ImageCache(index.root, config.global_cache_frames)
        self.progress = progress or (lambda value: None)
        self.attempts = 0
        self.failures = 0
        self.committed = 0
        self.stop_after = None

    def audit(self, unit, chunk, payload):
        self.attempts += 1
        path = (
            self.store.root
            / "audit"
            / f"s{unit.session}_w{unit.window}_t{unit.track}_{chunk.start}-{chunk.stop}_{self.attempts}_{time.time_ns()}.json"
        )
        atomic_json(path, payload)

    def process(self, unit, chunk):
        if self.stop_after is not None and self.committed >= self.stop_after:
            return
        owned = [f"{unit.session}:{unit.window}:{unit.track}:{f}" for f in chunk.primary]
        pending = [f for f, k in zip(chunk.primary, owned) if not self.store.contains(k)]
        if not pending:
            return
        # A crash after one recursively split half must not overwrite that half on resume.
        if len(pending) != len(owned):
            for start, stop in contiguous(pending):
                self.process(
                    unit, type(chunk)(start, stop, chunk.window_start, chunk.window_stop, chunk.context)
                )
            return
        started = time.monotonic()
        try:
            sources = self.index.frames(unit, chunk.inputs)
            prepared = []
            for source in sources:
                crop = self.images.load(source["crop_source"])
                global_image = self.images.load(source["global_source"], cached=True)
                location = self.localiser.locate(crop, global_image, unit.key)
                prepared.append(
                    {
                        "source": source,
                        "crop": crop,
                        "global": highlight(global_image, location),
                        "localisation": location,
                    }
                )
            primary_sources = [
                item["source"] for item in prepared if item["source"]["source_frame"] in chunk.primary
            ]
            visible_sources = [s for s in primary_sources if s["crop_source"] is not None]
            records = [missing_record(s) for s in primary_sources if s["crop_source"] is None]
            if visible_sources:
                primary = [s["source_frame"] for s in visible_sources]
                locations = {item["source"]["source_frame"]: item["localisation"] for item in prepared}
                allowed = {item["source"]["source_frame"] for item in prepared if item["crop"] is not None}
                correction = None
                for attempt in range(2):
                    raw = self.backend.generate(messages(prepared, primary, self.config, correction))
                    try:
                        annotations = expand_response(raw, visible_sources, allowed, locations)
                    except ValueError as exc:
                        self.audit(
                            unit,
                            chunk,
                            {
                                "type": "invalid_output",
                                "attempt": attempt + 1,
                                "reason": str(exc),
                                "raw_response": raw,
                            },
                        )
                        if attempt:
                            if isinstance(exc, PlaceholderEvidenceError):
                                raise PlaceholderEvidenceError(str(exc)) from exc
                            raise SplitRequired(
                                "Model output failed validation after one correction retry: "
                                + str(exc)[:1200]
                            ) from exc
                        correction = str(exc)
                    else:
                        self.audit(
                            unit,
                            chunk,
                            {
                                "type": "accepted_output",
                                "attempt": attempt + 1,
                                "raw_response": raw,
                                "response_normalisation": "single_json_fence_removed"
                                if model_json_text(raw) != raw.strip()
                                else "none",
                                "localisation": locations,
                                "metrics": getattr(self.backend, "last_metrics", {}),
                            },
                        )
                        records.extend(annotations)
                        break
            records.sort(key=lambda r: r["source_frame"])
            if [r["source_frame"] for r in records] != list(chunk.primary):
                raise ValueError("Incomplete primary ownership; chunk cannot be committed")
            chunk_id = f"{unit.key}/{chunk.start}-{chunk.stop}"
            self.store.commit(chunk_id, records)
            self.committed += 1
            metrics = {
                "chunk": chunk_id,
                "frames": len(records),
                "seconds": time.monotonic() - started,
                **getattr(self.backend, "last_metrics", {}),
            }
            with (self.store.root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical(metrics) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.progress(f"Committed {chunk_id}: {len(records)} positions")
        except SplitRequired as exc:
            self.audit(unit, chunk, {"type": "split_required", "reason": str(exc)})
            # CUDA exception tracebacks can retain failed-call tensor references.
            reason = str(exc)
            exc.__traceback__ = exc.__cause__ = exc.__context__ = None
            if hasattr(self.backend, "release_memory"):
                self.backend.release_memory()
            if chunk.stop - chunk.start <= 1:
                self.failures += 1
                self.progress(f"Incomplete single frame {unit.key}/{chunk.start}: {reason}")
            else:
                for child in chunk.split():
                    self.process(unit, child)
        except FatalGPUError:
            raise
        except PlaceholderEvidenceError as exc:
            self.audit(unit, chunk, {"type": "ungrounded_output", "reason": str(exc)})
            raise RuntimeError(
                "Repeated placeholder evidence after a correction retry; stopping without accepting invented observations"
            ) from exc
        except (SourceError, ValueError) as exc:
            self.failures += 1
            self.audit(unit, chunk, {"type": "incomplete", "reason": str(exc)})
            self.progress(f"Incomplete {unit.key}/{chunk.start}-{chunk.stop}: {exc}")

    def run(self, units, max_chunks=None):
        self.stop_after = max_chunks
        iterator = task_iterator(units, self.config)
        tasks = self.backend.iter_tasks(iterator) if hasattr(self.backend, "iter_tasks") else iterator
        for unit, chunk in tasks:
            if max_chunks is not None and self.committed >= max_chunks:
                break
            self.process(unit, chunk)
        return {
            **self.store.counts(),
            "chunks_committed_this_invocation": self.committed,
            "failed_chunks_this_invocation": self.failures,
        }


def contiguous(frames):
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            yield start, previous + 1
            start = frame
        previous = frame
    yield start, previous + 1


def run_annotation(index, config, units, run_dir: Path, model_dir: Path, max_chunks=None, progress=None):
    run_dir = outside(run_dir, index.root)
    index.verify_metadata(config)
    identity = model_identity(model_dir, config)
    manifest = run_manifest(index, config, units, identity)
    with writer_lock(run_dir / ".writer.lock"):
        store = RunStore(run_dir, manifest)
        try:
            if store.counts()["complete"]:
                return store.counts()
            backend = QwenBackend(config, model_dir)
            runner = Runner(index, config, store, backend, Localiser(config), progress)
            return runner.run(units, max_chunks)
        finally:
            store.close()


def dry_run(index, config, units, run_dir: Path) -> dict:
    run_dir = outside(run_dir, index.root)
    index.verify_metadata(config)
    result = {
        "kind": "inference_plan_only",
        "annotations_generated": False,
        "dataset_fingerprint": index.manifest["dataset_fingerprint"],
        "config_fingerprint": config.fingerprint,
        "track_windows": len(units),
        "expected_positions": len(units) * config.window_frames,
        "primary_chunks": 0,
        "input_image_occurrences_with_overlap": 0,
        "missing_crop_positions": 0,
        "missing_global_positions": 0,
        "first_chunk": None,
    }
    for unit, chunk in task_iterator(units, config):
        frames = index.frames(unit, chunk.inputs)
        result["primary_chunks"] += 1
        result["input_image_occurrences_with_overlap"] += sum(
            int(f["crop_source"] is not None) + int(f["global_source"] is not None) for f in frames
        )
        primary = [f for f in frames if f["source_frame"] in chunk.primary]
        result["missing_crop_positions"] += sum(f["crop_source"] is None for f in primary)
        result["missing_global_positions"] += sum(f["global_source"] is None for f in primary)
        if result["first_chunk"] is None:
            result["first_chunk"] = {
                "unit": unit.key,
                "primary": list(chunk.primary),
                "input_frames": list(chunk.inputs),
                "sources": frames,
            }
    atomic_json(run_dir / "inference_plan.json", result)
    return result
