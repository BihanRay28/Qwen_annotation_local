import json
import sqlite3
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from PIL import Image

from cue_annotation.cli import dispatch, parser
from cue_annotation.config import Config
from cue_annotation.dataset import Unit
from cue_annotation.pilots import plan_pilots, selected_units
from cue_annotation.qwen_backend import resolve_cached_model
from cue_annotation.util import atomic_json, digest


class SelectionIndex:
    """Bounded in-memory index fixture for selection tests, never an inference backend."""

    def __init__(self, root, windows=20):
        self.root = root
        self.manifest = {"dataset_fingerprint": "fixture"}
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE files(session INT,window INT,track INT,frame INT,path TEXT)")
        Image.new("RGB", (100, 100)).save(root / "global.jpg")
        self._units = []
        for number in range(windows):
            session, window = (5 if number < 10 else 7), number % 10 + 1
            for frame in range(250):
                self.conn.execute(
                    "INSERT INTO files VALUES(?,?,?,?,?)", (session, window, -1, frame, "global.jpg")
                )
            for track, count in ((2, 250), (3, 180), (4, 100)):
                name = f"crop{track}.jpg"
                Image.new("RGB", (track * 4, track * 5)).save(root / name)
                self._units.append(Unit(session, window, track, (window - 1) * 250))
                self.conn.executemany(
                    "INSERT INTO files VALUES(?,?,?,?,?)",
                    [(session, window, track, f, name) for f in range(count)],
                )
        self.conn.commit()

    def units(self):
        return self._units

    def verify_metadata(self, config):
        pass


class PilotSelectionTests(unittest.TestCase):
    def test_distinct_window_selection_and_frozen_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "dataset"
            root.mkdir()
            index = SelectionIndex(root)
            try:
                output = base / "selection.json"
                plan = plan_pilots(index, Config(), output)
                self.assertEqual(len(plan["stages"]["one"]["units"]), 1)
                self.assertEqual(len(plan["stages"]["five"]["units"]), 5)
                self.assertEqual(len(plan["stages"]["twenty"]["units"]), 60)
                self.assertEqual(plan["stages"]["twenty"]["classroom_windows"], 20)
                self.assertFalse(plan["quality_reviewed"])
                five = selected_units(output, "five", index, Config())
                self.assertEqual(len({(u.session, u.window) for u in five}), 5)
                self.assertEqual({u.session for u in five}, {5, 7})
                self.assertEqual(plan_pilots(index, Config(), output), plan)
                with self.assertRaisesRegex(ValueError, "different track count"):
                    plan_pilots(index, Config(), output, 2)
                with self.assertRaisesRegex(ValueError, "different dataset/configuration"):
                    selected_units(output, "one", index, Config(primary_frames=10))
                plan["stages"]["one"]["units"].append(plan["stages"]["one"]["units"][0])
                plan.pop("plan_fingerprint")
                plan["plan_fingerprint"] = digest(plan)
                atomic_json(output, plan)
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    selected_units(output, "one", index, Config())
            finally:
                index.conn.close()

    def test_insufficient_inventory_does_not_claim_twenty_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "dataset"
            root.mkdir()
            index = SelectionIndex(root, 19)
            try:
                with self.assertRaisesRegex(ValueError, "found 19"):
                    plan_pilots(index, Config(), base / "selection.json")
                self.assertFalse((base / "selection.json").exists())
            finally:
                index.conn.close()

    def test_tampered_plan_and_unknown_units_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "dataset"
            root.mkdir()
            index = SelectionIndex(root)
            try:
                output = base / "selection.json"
                plan = plan_pilots(index, Config(), output)
                plan["stages"]["one"]["units"][0]["track"] = 999
                atomic_json(output, plan)
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    selected_units(output, "one", index, Config())
                plan.pop("plan_fingerprint")
                plan["plan_fingerprint"] = digest(plan)
                atomic_json(output, plan)
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    selected_units(output, "one", index, Config())
            finally:
                index.conn.close()

    def test_cli_uses_exact_units_without_cross_product(self):
        config = Config()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            from helpers import dataset

            root, _, index, units, _ = dataset(base)
            try:
                plan = {
                    "version": 1,
                    "dataset_fingerprint": index.manifest["dataset_fingerprint"],
                    "config_fingerprint": config.fingerprint,
                    "stages": {"one": {"units": [asdict(units[0])]}},
                }
                plan["plan_fingerprint"] = digest(plan)
                selection = base / "selection.json"
                atomic_json(selection, plan)
                arguments = [
                    "run",
                    "--dataset-root",
                    str(root),
                    "--index",
                    str(index.path),
                    "--run-dir",
                    str(base / "dry"),
                    "--units-file",
                    str(selection),
                    "--pilot-stage",
                    "one",
                    "--dry-run",
                ]
                self.assertEqual(dispatch(parser().parse_args(arguments)), 0)
                report = json.loads((base / "dry/inference_plan.json").read_text())
                self.assertEqual(report["expected_positions"], 250)
                with self.assertRaisesRegex(ValueError, "other selection flags"):
                    dispatch(parser().parse_args(arguments + ["--session", "5"]))
            finally:
                index.close()


class OfflineModelResolutionTests(unittest.TestCase):
    def test_immutable_download_without_main_ref_is_resolved_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            repository = cache / "models--Qwen--Qwen2.5-VL-7B-Instruct"
            snapshot = repository / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)
            atomic_json(
                repository / "eascca_download_manifest.json",
                {
                    "model_id": Config().model_id,
                    "requested_revision": "main",
                    "resolved_revision": snapshot.name,
                },
            )
            hub = ModuleType("huggingface_hub")
            constants = ModuleType("huggingface_hub.constants")
            constants.HF_HUB_CACHE = str(cache)
            hub.snapshot_download = lambda *a, **kw: self.fail("Must not resolve missing mutable main ref")
            with patch.dict("sys.modules", {"huggingface_hub": hub, "huggingface_hub.constants": constants}):
                self.assertEqual(resolve_cached_model(Config()), snapshot)
                atomic_json(
                    repository / "eascca_download_manifest.json",
                    {
                        "model_id": Config().model_id,
                        "requested_revision": "main",
                        "resolved_revision": "../../escape",
                    },
                )
                with self.assertRaisesRegex(ValueError, "Invalid cached"):
                    resolve_cached_model(Config())
