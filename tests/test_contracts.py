import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import dataset, image, model_frame

from cue_annotation.chunking import chunks
from cue_annotation.config import Config, load_config
from cue_annotation.dataset import build_index, image_files, preflight
from cue_annotation.prompts import messages
from cue_annotation.schema import expand_response, missing_record, validate_record
from cue_annotation.util import outside


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_chunk_primary_ownership_and_context(self):
        values = list(chunks(9225))
        self.assertEqual([f for c in values for f in c.primary], list(range(9225, 9475)))
        self.assertEqual(len(values), 10)
        self.assertEqual(len(values[1].inputs), 35)
        self.assertEqual(min(values[0].inputs), 9225)
        self.assertEqual(max(values[-1].inputs), 9474)
        left, right = values[1].split()
        self.assertEqual(list(left.primary) + list(right.primary), list(values[1].primary))
        self.assertFalse(set(left.primary) & set(right.primary))

    def test_invalid_config_rejects_sampling_and_bool(self):
        path = self.base / "config.json"
        path.write_text('{"sample_every": 2}')
        with self.assertRaises(ValueError):
            load_config(path)
        with self.assertRaises(ValueError):
            Config(primary_frames=True).validate()
        with self.assertRaises(ValueError):
            Config(num_workers=2).validate()

    def test_output_cannot_be_inside_dataset(self):
        root = self.base / "dataset"
        with self.assertRaises(ValueError):
            outside(root / "index.sqlite", root)

    def test_matching_is_exact_and_start_is_not_inferred(self):
        root = self.base / "dataset"
        window = root / "session7/window1"
        image(window / "global_frames/frame_009226.jpg")
        image(window / "cropped_frames_per_student/track_2/frame_009227.jpg")
        build_index(root, self.base / "index.sqlite", Config())
        from cue_annotation.dataset import DatasetIndex

        index = DatasetIndex(self.base / "index.sqlite", root)
        try:
            rows = index.frames(index.units()[0], range(9225, 9228))
            self.assertEqual(index.units()[0].start, 9225)
            self.assertEqual(
                [r["input_status"] for r in rows], ["both_missing", "crop_missing", "global_missing"]
            )
        finally:
            index.close()

    def test_numeric_sorting_and_track_identity(self):
        root, config, index, _, _ = dataset(self.base, session=5, window=10, track=10)
        index.close()
        for window, track in [(2, 10), (2, 2)]:
            image(root / f"session5/window{window}/global_frames/frame_{(window - 1) * 250:06d}.jpg")
            image(
                root
                / f"session5/window{window}/cropped_frames_per_student/track_{track}/frame_{(window - 1) * 250:06d}.jpg"
            )
        build_index(root, self.base / "index2.sqlite", config)
        from cue_annotation.dataset import DatasetIndex

        index = DatasetIndex(self.base / "index2.sqlite", root)
        try:
            self.assertEqual([(u.window, u.track) for u in index.units()], [(2, 2), (2, 10), (10, 10)])
        finally:
            index.close()

    def test_duplicate_frame_numbers_rejected(self):
        image(self.base / "frame_000001.jpg")
        image(self.base / "frame_1.jpeg")
        with self.assertRaises(ValueError):
            image_files(self.base)

    def test_partial_crop_root_blocks_index(self):
        root, config, index, _, _ = dataset(self.base)
        index.close()
        (root / "session5/window1/cropped_frames_per_student.__partial").mkdir()
        with self.assertRaises(ValueError):
            build_index(root, self.base / "other.sqlite", config)

    def test_out_of_range_frames_block_index(self):
        root, config, index, _, _ = dataset(self.base)
        index.close()
        image(root / "session5/window1/cropped_frames_per_student/track_2/frame_000250.jpg")
        with self.assertRaises(ValueError):
            build_index(root, self.base / "other.sqlite", config)

    def test_source_hash_detects_change_with_preserved_metadata(self):
        root, _, index, _, _ = dataset(self.base)
        path = root / "session5/window1/global_frames/frame_000000.jpg"
        stat = path.stat()
        content = bytearray(path.read_bytes())
        content[-2] ^= 1
        path.write_bytes(content)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        try:
            with self.assertRaises(ValueError):
                index.verify_sources()
        finally:
            index.close()

    def test_subset_never_claims_full_inventory(self):
        root, config, index, _, _ = dataset(self.base)
        index.close()
        summary = preflight(root, config)
        self.assertFalse(summary["historical_full_inventory"])
        self.assertEqual(summary["missing_crop_positions"], 247)
        self.assertIn("7", summary["missing_windows_against_historical_scope"])

    def source(self):
        root, _, index, units, _ = dataset(self.base)
        try:
            return index.frames(units[0], [0])[0]
        finally:
            index.close()

    def test_hidden_eyes_and_rejected_location_become_unknown(self):
        source = self.source()
        frame = model_frame(0)
        frame.update(
            present=["EYES_CLOSED", "LOOK_PEER"],
            unknown=[],
            evidence=[{"frames": [0], "description": "Synthetic fixture"}],
        )
        record = expand_response(json.dumps({"frames": [frame]}), [source], {0}, {0: {"status": "rejected"}})[
            0
        ]
        self.assertEqual(record["cues"]["EYES_CLOSED"], "unknown")
        self.assertEqual(record["cues"]["LOOK_PEER"], "unknown")
        self.assertIn("weak_localisation", record["review_flags"])

    def test_wrong_or_duplicate_primary_frames_rejected(self):
        source = self.source()
        for frames in [[model_frame(1)], [model_frame(0), model_frame(0)]]:
            with self.assertRaises(ValueError):
                expand_response(json.dumps({"frames": frames}), [source], {0}, {0: {"status": "accepted"}})

    def test_bad_cue_and_outside_evidence_rejected(self):
        source = self.source()
        frame = model_frame(0)
        frame["present"] = ["ENGAGED"]
        with self.assertRaises(ValueError):
            expand_response(json.dumps({"frames": [frame]}), [source], {0}, {0: {"status": "accepted"}})
        frame = model_frame(0)
        frame["evidence"] = [{"frames": [99], "description": "Not supplied"}]
        with self.assertRaises(ValueError):
            expand_response(json.dumps({"frames": [frame]}), [source], {0}, {0: {"status": "accepted"}})

    def test_present_requires_current_frame_evidence(self):
        source = self.source()
        frame = model_frame(0)
        frame.update(
            present=["HEAD_DOWN"], unknown=[], evidence=[{"frames": [1], "description": "Other frame"}]
        )
        with self.assertRaises(ValueError):
            expand_response(json.dumps({"frames": [frame]}), [source], {0, 1}, {0: {"status": "accepted"}})

    def test_missing_crop_is_all_unknown_not_unobservable(self):
        source = self.source()
        source["crop_source"] = None
        source["input_status"] = "crop_missing"
        record = missing_record(source)
        self.assertEqual(set(record["cues"].values()), {"unknown"})
        record["cues"]["STUDENT_NOT_OBSERVABLE"] = "present"
        with self.assertRaises(ValueError):
            validate_record(record)

    def test_input_status_and_time_cannot_be_forged(self):
        source = self.source()
        source["crop_source"] = source["global_source"] = None
        source["input_status"] = "both_missing"
        record = missing_record(source)
        record["time_seconds"] = 8
        with self.assertRaises(ValueError):
            validate_record(record)
        record["time_seconds"] = 0
        record["input_status"] = "matched"
        with self.assertRaises(ValueError):
            validate_record(record)

    def test_prompt_only_requests_owned_positions(self):
        prepared = [
            {
                "source": {"source_frame": 1, "input_status": "both_missing"},
                "crop": None,
                "global": None,
                "localisation": {"status": "unavailable"},
            }
        ]
        content = messages(prepared, [2], Config())[-1]["content"]
        self.assertIn("CONTEXT ONLY", content[1]["text"])
        self.assertIn("once each: 2", content[-1]["text"])

    def test_core_import_does_not_load_torch(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import cue_annotation.cli, cue_annotation.runner; import sys; "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules",
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
