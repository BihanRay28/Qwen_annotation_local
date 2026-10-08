import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from helpers import FakeBackend, FakeLocaliser, dataset

from cue_annotation.export import export_run, summarise
from cue_annotation.ontology import CUES
from cue_annotation.persistence import RunStore, inspect_run, writer_lock
from cue_annotation.runner import Runner, dry_run
from cue_annotation.schema import missing_record


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root, self.config, self.index, self.units, self.manifest = dataset(self.base, count=25)
        self.run = self.base / "run"

    def tearDown(self):
        self.index.close()
        self.temporary.cleanup()

    def missing(self, frame=0):
        source = self.index.frames(self.units[0], [frame])[0]
        source["global_source"] = source["crop_source"] = None
        source["input_status"] = "both_missing"
        return missing_record(source)

    def test_commit_flush_and_rebuild_from_journal(self):
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            store.commit("first", [self.missing()])
            store.close()
        (self.run / "state.sqlite").write_bytes(b"broken derived database")
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            self.assertEqual(store.counts()["committed_frames"], 1)
            store.close()
        self.assertEqual(inspect_run(self.run)["committed_frames"], 1)

    def test_interrupted_tail_is_preserved_and_recovered(self):
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            store.commit("first", [self.missing()])
            store.close()
        journal = self.run / "journal.jsonl"
        with journal.open("ab") as stream:
            stream.write(b'{"type":"unfinished')
        with self.assertRaises(ValueError):
            inspect_run(self.run)
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            self.assertEqual(store.counts()["committed_frames"], 1)
            store.close()
        self.assertTrue(list(self.run.glob("journal.jsonl.interrupted-*")))
        self.assertEqual(inspect_run(self.run)["remaining_frames"], 249)

    def test_duplicate_and_out_of_scope_records_rejected(self):
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            try:
                store.commit("first", [self.missing()])
                with self.assertRaises(ValueError):
                    store.commit("duplicate", [self.missing()])
                record = self.missing()
                record["track"] = 999
                with self.assertRaises(ValueError):
                    store.commit("other", [record])
            finally:
                store.close()

    def test_changed_manifest_rejected(self):
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            store.close()
        changed = deepcopy(self.manifest)
        changed["config"]["primary_frames"] = 1
        with self.assertRaises(ValueError):
            RunStore(self.run, changed)

    def test_corrupt_committed_line_is_not_discarded(self):
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            store.commit("first", [self.missing()])
            store.close()
        journal = self.run / "journal.jsonl"
        record = json.loads(journal.read_text())
        record["digest"] = "not-the-committed-digest"
        journal.write_text(json.dumps(record) + "\n")
        with self.assertRaises(ValueError):
            RunStore(self.run, self.manifest)
        self.assertTrue(journal.exists())

    def test_writer_lock_releases_after_process_crash(self):
        lock = self.run / ".writer.lock"
        script = "import os; from pathlib import Path; from cue_annotation.persistence import writer_lock; "
        script += f"\nwith writer_lock(Path({str(lock)!r})):\n os._exit(0)"
        subprocess.run([sys.executable, "-c", script], check=True)
        with writer_lock(lock):
            pass

    def test_concurrent_writer_rejected(self):
        lock = self.run / ".writer.lock"
        with writer_lock(lock):
            script = "from pathlib import Path; from cue_annotation.persistence import writer_lock; "
            script += f"\nwith writer_lock(Path({str(lock)!r})):\n pass"
            process = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Another process", process.stderr)

    def test_recursive_split_resume_preserves_accepted_half(self):
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            first = Runner(self.index, self.config, store, FakeBackend(limit=13), FakeLocaliser())
            first.run(self.units, max_chunks=1)
            self.assertEqual(store.counts()["committed_frames"], 12)
            journal_before = (self.run / "journal.jsonl").read_bytes()
            store.close()
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            second = Runner(self.index, self.config, store, FakeBackend(limit=13), FakeLocaliser())
            result = second.run(self.units)
            self.assertTrue(result["complete"])
            store.close()
        self.assertTrue((self.run / "journal.jsonl").read_bytes().startswith(journal_before))
        self.assertEqual(inspect_run(self.run)["committed_frames"], 250)

    def test_failed_model_response_stays_incomplete(self):
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            backend = FakeBackend(invalid=True)
            runner = Runner(self.index, self.config, store, backend, FakeLocaliser())
            result = runner.run(self.units)
            self.assertFalse(result["complete"])
            self.assertEqual(result["committed_frames"], 225)
            self.assertEqual(backend.calls, 2)
            self.assertTrue(list((self.run / "audit").glob("*.json")))
            store.close()

    def test_dry_run_has_no_journal_or_labels(self):
        result = dry_run(self.index, self.config, self.units, self.run)
        self.assertFalse(result["annotations_generated"])
        self.assertEqual(result["expected_positions"], 250)
        self.assertFalse((self.run / "journal.jsonl").exists())

    def test_export_records_are_sorted_and_review_is_local(self):
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            Runner(self.index, self.config, store, FakeBackend(), FakeLocaliser()).run(self.units)
            store.close()
        destination = self.base / "export"
        result = export_run(self.run, destination, self.root, review_images=2)
        self.assertTrue(result["complete"])
        records = [json.loads(line) for line in (destination / "frames.jsonl").read_text().splitlines()]
        self.assertEqual([r["source_frame"] for r in records], list(range(250)))
        self.assertTrue((destination / "review.html").exists())
        self.assertEqual(len(list((destination / "review_images").glob("*.png"))), 2)

    def test_unknown_breaks_supported_interval(self):
        with writer_lock(self.run / ".writer.lock"):
            store = RunStore(self.run, self.manifest)
            Runner(self.index, self.config, store, FakeBackend(), FakeLocaliser()).run(
                self.units, max_chunks=1
            )
            store.close()
        event = json.loads((self.run / "journal.jsonl").read_text().splitlines()[0])
        records = event["records"][:3]
        records[0]["cues"]["HEAD_DOWN"] = "present"
        records[2]["cues"]["HEAD_DOWN"] = "present"
        result = summarise(records)
        self.assertEqual(len(result["supported_intervals"]["HEAD_DOWN"]), 2)
        self.assertEqual(result["supported_intervals"]["HEAD_DOWN"][0]["duration_seconds"], 0.04)
        self.assertEqual(set(result["cue_counts"]), set(CUES))


if __name__ == "__main__":
    unittest.main()
