import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from helpers import FakeBackend, FakeLocaliser, dataset, model_frame

from cue_annotation.persistence import RunStore, writer_lock
from cue_annotation.prompts import CONTRACT_PROMPT
from cue_annotation.runner import Runner
from cue_annotation.schema import PlaceholderEvidenceError, expand_response, missing_record

spec = importlib.util.spec_from_file_location(
    "archive_empty_pilots", Path(__file__).resolve().parents[1] / "scripts/archive_empty_pilots.py"
)
archive_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_module)


class ResponseTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root, self.config, self.index, self.units, self.manifest = dataset(self.base, count=2)
        self.sources = self.index.frames(self.units[0], [0, 1])
        self.locations = {
            f: {"status": "accepted", "bbox": [0, 0, 4, 4], "score": 1.0, "margin": 1.0} for f in [0, 1]
        }
        self.raw = json.dumps({"frames": [model_frame(f) for f in [0, 1]]})

    def tearDown(self):
        self.index.close()
        self.temporary.cleanup()

    def expand(self, raw):
        return expand_response(raw, self.sources, {0, 1}, self.locations)

    def test_single_complete_json_fence_preserves_exact_records(self):
        bare = self.expand(self.raw)
        for opening in ["```json", "```JSON", "```"]:
            self.assertEqual(self.expand(opening + "\n" + self.raw + "\n```"), bare)

    def test_surrounding_prose_multiple_objects_and_partial_json_stay_invalid(self):
        for raw in [
            "Here is the answer:\n```json\n" + self.raw + "\n```",
            "```json\n" + self.raw + "\n```\nDone.",
            self.raw + self.raw,
            "```json\n" + self.raw[:-1] + "\n```",
        ]:
            with self.assertRaises(ValueError):
                self.expand(raw)

    def test_template_evidence_is_rejected_even_when_all_frame_ids_are_correct(self):
        frames = [model_frame(f) for f in [0, 1]]
        for frame in frames:
            frame["evidence"] = [
                {"frames": [frame["frame"]], "description": "Concrete visible evidence for this target"}
            ]
        with self.assertRaises(PlaceholderEvidenceError):
            self.expand("```json\n" + json.dumps({"frames": frames}) + "\n```")
        self.assertNotIn("Concrete visible evidence for this target", CONTRACT_PROMPT)
        self.assertNotIn('"present":["HEAD_DOWN"]', CONTRACT_PROMPT)

    def test_repeated_specific_observation_is_not_rejected_just_for_repetition(self):
        frames = [model_frame(f) for f in [0, 1]]
        for frame in frames:
            frame["unknown"].remove("HEAD_DOWN")
            frame["present"] = ["HEAD_DOWN"]
            frame["evidence"] = [
                {"frames": [frame["frame"]], "description": "The head is lowered toward the desk."}
            ]
        records = self.expand(json.dumps({"frames": frames}))
        self.assertTrue(all(r["cues"]["HEAD_DOWN"] == "present" for r in records))

    def test_structurally_oversized_response_splits_without_losing_frames(self):
        class ShortResponseBackend(FakeBackend):
            def generate(self, messages):
                raw = json.loads(super().generate(messages))
                raw["frames"] = raw["frames"][:1]
                return "```json\n" + json.dumps(raw) + "\n```"

        run = self.base / "run"
        with writer_lock(run / ".writer.lock"):
            store = RunStore(run, self.manifest)
            try:
                result = Runner(self.index, self.config, store, ShortResponseBackend(), FakeLocaliser()).run(
                    self.units
                )
                self.assertTrue(result["complete"])
                self.assertEqual(result["committed_frames"], 250)
                self.assertTrue(
                    any(
                        json.loads(p.read_text())["type"] == "split_required"
                        for p in (run / "audit").glob("*.json")
                    )
                )
                self.assertTrue(
                    any(
                        json.loads(p.read_text()).get("response_normalisation") == "single_json_fence_removed"
                        for p in (run / "audit").glob("*.json")
                    )
                )
            finally:
                store.close()

    def test_persistent_template_copy_stops_after_one_retry_without_any_commit(self):
        class TemplateBackend(FakeBackend):
            def generate(self, messages):
                raw = json.loads(super().generate(messages))
                for frame in raw["frames"]:
                    frame["evidence"] = [
                        {
                            "frames": [frame["frame"]],
                            "description": "Concrete visible evidence for this target",
                        }
                    ]
                return json.dumps(raw)

        backend = TemplateBackend()
        run = self.base / "run"
        with writer_lock(run / ".writer.lock"):
            store = RunStore(run, self.manifest)
            try:
                with self.assertRaisesRegex(RuntimeError, "Repeated placeholder"):
                    Runner(self.index, self.config, store, backend, FakeLocaliser()).run(self.units)
                self.assertEqual(backend.calls, 2)
                self.assertEqual(store.counts()["committed_frames"], 0)
            finally:
                store.close()


class EmptyPilotArchiveTests(unittest.TestCase):
    def test_zero_record_run_and_raw_audits_are_preserved_with_index_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, _, index, _, manifest = dataset(base)
            index.close()
            suite = base / "suite"
            run = suite / "pilot-one"
            with writer_lock(run / ".writer.lock"):
                store = RunStore(run, manifest)
                store.close()
            (run / "audit").mkdir()
            raw = b'{"type":"invalid_output","raw_response":"synthetic rejected response"}'
            (run / "audit/rejected.json").write_bytes(raw)
            (suite / "index.sqlite").write_bytes(b"retained source index")
            saved = archive_module.archive_empty_pilots(suite, "previous-revision")
            self.assertFalse(run.exists())
            self.assertEqual((saved / "pilot-one/audit/rejected.json").read_bytes(), raw)
            self.assertEqual((suite / "index.sqlite").read_bytes(), b"retained source index")
            self.assertEqual(json.loads((saved / "archive.json").read_text())["accepted_records"], 0)
            self.assertTrue(root.exists())

    def test_any_accepted_run_blocks_the_entire_archive_before_moving_empty_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, _, index, units, manifest = dataset(base)
            suite = base / "suite"
            try:
                for name in ["pilot-one", "pilot-five"]:
                    run = suite / name
                    with writer_lock(run / ".writer.lock"):
                        store = RunStore(run, manifest)
                        if name == "pilot-five":
                            source = index.frames(units[0], [10])[0]
                            store.commit("accepted", [missing_record(source)])
                        store.close()
                with self.assertRaisesRegex(RuntimeError, "nonempty journal"):
                    archive_module.archive_empty_pilots(suite, "old")
                self.assertTrue((suite / "pilot-one/manifest.json").exists())
                self.assertTrue((suite / "pilot-five/journal.jsonl").exists())
                self.assertFalse((suite / "failed-empty-pilots").exists())
            finally:
                index.close()

    def test_active_pilot_writer_blocks_archiving(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            run = suite / "pilot-one"
            with writer_lock(run / ".writer.lock"):
                (run / "manifest.json").write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "still active"):
                    archive_module.archive_empty_pilots(suite, "old")
            self.assertTrue((run / "manifest.json").exists())

    def test_derived_accepted_records_block_archive_even_if_journal_was_lost(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, _, index, units, manifest = dataset(base)
            suite = base / "suite"
            run = suite / "pilot-one"
            try:
                with writer_lock(run / ".writer.lock"):
                    store = RunStore(run, manifest)
                    store.commit("accepted", [missing_record(index.frames(units[0], [10])[0])])
                    store.close()
                (run / "journal.jsonl").unlink()
                with self.assertRaisesRegex(RuntimeError, "accepted records exist"):
                    archive_module.archive_empty_pilots(suite, "old")
                self.assertTrue((run / "state.sqlite").exists())
                self.assertFalse((suite / "failed-empty-pilots").exists())
            finally:
                index.close()
