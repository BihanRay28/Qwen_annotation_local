import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cue_annotation.config import Config
from cue_annotation.evaluation import evaluate
from cue_annotation.localisation import Localiser, highlight
from cue_annotation.ontology import CUES
from cue_annotation.schema import missing_record


class LocalisationTests(unittest.TestCase):
    def test_native_alternatives_remain_ambiguous_after_coarse_resizing(self):
        random = np.random.default_rng(7)
        scene = random.integers(0, 256, (1000, 1600, 3), dtype=np.uint8)
        target = random.integers(0, 256, (90, 60, 3), dtype=np.uint8)
        scene[41:131, 71:131] = target
        scene[605:695, 1007:1067] = target
        result = Localiser(Config(localisation_scales=[1.0])).locate(
            Image.fromarray(target), Image.fromarray(scene), "track"
        )
        self.assertEqual(result["status"], "rejected")

    def test_exact_unique_native_match(self):
        random = np.random.default_rng(21)
        array = random.integers(0, 256, (200, 300, 3), dtype=np.uint8)
        scene = Image.fromarray(array)
        crop = scene.crop((70, 40, 110, 100))
        localiser = Localiser(Config(localisation_scales=[1.0]))
        result = localiser.locate(crop, scene, "track")
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["bbox"], [70, 40, 110, 100])
        original = scene.tobytes()
        outlined = highlight(scene, result)
        self.assertNotEqual(outlined.tobytes(), original)
        self.assertEqual(scene.tobytes(), original)

    def test_duplicate_target_match_is_rejected(self):
        random = np.random.default_rng(11)
        scene = random.integers(0, 256, (200, 300, 3), dtype=np.uint8)
        target = random.integers(0, 256, (40, 30, 3), dtype=np.uint8)
        scene[20:60, 20:50] = target
        scene[100:140, 200:230] = target
        result = Localiser(Config(localisation_scales=[1.0])).locate(
            Image.fromarray(target), Image.fromarray(scene), "track"
        )
        self.assertEqual(result["status"], "rejected")

    def test_low_texture_and_missing_inputs_abstain(self):
        localiser = Localiser(Config())
        image = Image.new("RGB", (50, 50), "white")
        self.assertEqual(localiser.locate(image, image, "track")["status"], "rejected")
        self.assertEqual(localiser.locate(None, image, "track")["status"], "unavailable")


class EvaluationTests(unittest.TestCase):
    def test_abstention_on_visible_positive_is_missed_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "session": 5,
                "window": 1,
                "track": 2,
                "source_frame": 0,
                "time_seconds": 0.0,
                "global_source": None,
                "crop_source": None,
                "input_status": "both_missing",
            }
            prediction = missing_record(source)
            reference = {
                "session": 5,
                "window": 1,
                "track": 2,
                "source_frame": 0,
                "adjudicated": True,
                "cues": {c: "unknown" for c in CUES},
            }
            reference["cues"]["HEAD_DOWN"] = "present"
            (root / "prediction.jsonl").write_text(json.dumps(prediction) + "\n")
            (root / "reference.jsonl").write_text(json.dumps(reference) + "\n")
            result = evaluate(root / "prediction.jsonl", root / "reference.jsonl")
            self.assertEqual(result["per_cue"]["HEAD_DOWN"]["fn"], 1)
            self.assertEqual(result["per_cue"]["HEAD_DOWN"]["recall"], 0)
            self.assertIsNone(result["per_cue"]["EYES_CLOSED"]["recall"])
            self.assertEqual(result["per_cue"]["HEAD_DOWN"]["answer_coverage"], 0)


if __name__ == "__main__":
    unittest.main()
