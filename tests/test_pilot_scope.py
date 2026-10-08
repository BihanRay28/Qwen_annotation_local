import json
import tempfile
import unittest
from pathlib import Path

from helpers import dataset, image

from cue_annotation.config import Config
from cue_annotation.dataset import DatasetIndex, build_index, preflight
from cue_annotation.pilot_scope import pilot_scope
from cue_annotation.util import atomic_json, digest


class PilotScopeTests(unittest.TestCase):
    def test_one_unfinished_window_is_excluded_without_altering_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, config, original, _, _ = dataset(base)
            original.close()
            incomplete = root / "session15/window88"
            start = config.session_start_frames["15"] + 87 * 250
            image(incomplete / f"global_frames/frame_{start}.jpg")
            (incomplete / "cropped_frames_per_student.__partial").mkdir()
            before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with self.assertRaisesRegex(ValueError, "unfinished"):
                build_index(root, base / "blocked.sqlite", config)
            result = build_index(root, base / "pilot.sqlite", config, excluded_windows=[(15, 88)])
            self.assertEqual(result["windows"], 1)
            self.assertEqual(result["excluded_windows"], [{"session": 15, "window": 88}])
            self.assertFalse(result["historical_full_inventory"])
            index = DatasetIndex(base / "pilot.sqlite", root)
            try:
                self.assertEqual({u.session for u in index.units()}, {5})
                self.assertEqual(index.manifest["selection"]["excluded_windows"], [[15, 88]])
                index.verify_metadata(config)
                image(incomplete / f"global_frames/frame_{start + 1}.jpg")
                index.verify_metadata(config)
            finally:
                index.close()
            for relative, data in before.items():
                self.assertEqual((root / relative).read_bytes(), data)
            self.assertTrue((incomplete / "cropped_frames_per_student.__partial").is_dir())

    def test_scope_records_exclusion_and_keeps_it_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            report = base / "report.json"
            output = base / "scope.json"
            atomic_json(
                report,
                {
                    "windows": 695,
                    "issues": [
                        {"session": 15, "window": 88, "issues": ["crop_root_not_finalised", "no_tracks"]}
                    ],
                },
            )
            scope = pilot_scope(report, output)
            self.assertEqual(scope["pilot_index_window_count"], 694)
            self.assertEqual(scope["exclusions"][0]["reason"], "crop_root_not_finalised")
            self.assertEqual(
                scope["scope_fingerprint"],
                digest({k: v for k, v in scope.items() if k != "scope_fingerprint"}),
            )
            atomic_json(report, {"windows": 695, "issues": []})
            self.assertEqual(pilot_scope(report, output), scope)
            atomic_json(
                report,
                {
                    "windows": 695,
                    "issues": [{"session": 5, "window": 1, "issues": ["crop_root_not_finalised"]}],
                },
            )
            with self.assertRaisesRegex(ValueError, "Additional windows"):
                pilot_scope(report, output)
            changed = json.loads(output.read_text())
            changed["exclusions"] = []
            atomic_json(output, changed)
            with self.assertRaisesRegex(ValueError, "modified"):
                pilot_scope(report, output)

    def test_out_of_range_frames_are_never_automatically_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            report = base / "report.json"
            atomic_json(
                report,
                {
                    "windows": 695,
                    "issues": [
                        {
                            "session": 15,
                            "window": 88,
                            "issues": ["crop_root_not_finalised", "out_of_range_global_frames:1"],
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "Out-of-range"):
                pilot_scope(report, base / "scope.json")
            self.assertFalse((base / "scope.json").exists())

    def test_invalid_exclusion_pairs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "positive integer"):
                preflight(Path(temporary), Config(), excluded_windows=[(15, False)])
