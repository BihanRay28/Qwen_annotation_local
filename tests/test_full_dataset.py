import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from helpers import FakeBackend, FakeLocaliser, dataset, image

from cue_annotation.cli import dispatch, parser
from cue_annotation.dataset import preflight
from cue_annotation.labels import LabelsSnapshot, full_scope
from cue_annotation.persistence import RunStore, writer_lock
from cue_annotation.runner import Runner, run_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_full_dataset import prepare  # noqa: E402
from publish_labels import publish  # noqa: E402


class FullDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_every_position_single_json_and_resume_from_journal(self):
        root, config, index, units, _ = dataset(self.base, count=3)
        config = replace(config, primary_frames=1, context_frames=2)
        manifest = run_manifest(
            index, config, units, {"model_id": "synthetic", "artifact_fingerprint": "test"}
        )
        inventory = preflight(root, config)
        run = self.base / "full/run"
        before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.jpg")}
        try:
            with writer_lock(run / ".writer.lock"):
                store = RunStore(run, manifest)
                snapshot = LabelsSnapshot(run, manifest, inventory, interval=999999)
                try:
                    result = Runner(
                        index, config, store, FakeBackend(), FakeLocaliser(), snapshot=snapshot
                    ).run(units, 2)
                    self.assertEqual(result["committed_frames"], 2)
                finally:
                    snapshot.close()
                    store.close()
            partial = json.loads((self.base / "full/labels.json").read_text())
            track = partial["sessions"]["session5"]["windows"]["window1"]["tracks"]["track_2"]
            self.assertEqual(list(track["frames"]), ["0", "1"])
            self.assertFalse(track["coverage"]["complete"])
            (run / "labels.sqlite").write_bytes(b"synthetic damaged derived sorting database")
            with writer_lock(run / ".writer.lock"):
                store = RunStore(run, manifest)
                snapshot = LabelsSnapshot(run, manifest, inventory)
                try:
                    result = Runner(
                        index, config, store, FakeBackend(), FakeLocaliser(), snapshot=snapshot
                    ).run(units)
                    self.assertTrue(result["complete"])
                finally:
                    snapshot.close()
                    store.close()
            publish(self.base / "full/labels.json", root, "labels.py")
            labels = json.loads((root / "labels.py").read_text())
            frames = labels["sessions"]["session5"]["windows"]["window1"]["tracks"]["track_2"]["frames"]
            self.assertEqual(list(frames), [str(f) for f in range(250)])
            self.assertEqual(frames["2"]["source_frame"], 2)
            self.assertEqual(set(frames["249"]["cues"].values()), {"unknown"})
            self.assertTrue(labels["metadata"]["indexed_annotations_complete"])
            self.assertFalse(labels["metadata"]["quality_reviewed"])
            self.assertEqual(len(list(run.glob("labels.corrupt-*.sqlite"))), 1)
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*.jpg")})
        finally:
            index.close()

    def test_unfinished_window_is_visible_in_json_without_invented_tracks(self):
        root, config, index, units, manifest = dataset(self.base)
        index.close()
        path = root / "session15/window88"
        image(path / "global_frames/frame_021750.jpg")
        (path / "cropped_frames_per_student.__partial").mkdir()
        report = preflight(root, config)
        self.assertEqual(full_scope(report), [(15, 88)])
        run = self.base / "full/run"
        store = RunStore(run, manifest)
        snapshot = LabelsSnapshot(run, manifest, report)
        snapshot.close()
        store.close()
        value = json.loads((self.base / "full/labels.json").read_text())
        window = value["sessions"]["session15"]["windows"]["window88"]
        self.assertIn("crop_root_not_finalised", window["source_issues"])
        self.assertEqual(window["tracks"], {})
        self.assertFalse(value["metadata"]["complete"])
        report["issues"][0]["issues"].append("out_of_range_crop_frames:track_2")
        with self.assertRaises(ValueError):
            full_scope(report)

    def test_bad_evidence_does_not_stop_full_attempt_or_become_an_accepted_label(self):
        class CopiesFirstFrame(FakeBackend):
            def generate(self, messages):
                value = json.loads(super().generate(messages))
                if value["frames"][0]["frame"] == 0:
                    value["frames"][0]["evidence"] = [
                        {"frames": [0], "description": "Concrete visible evidence for this target"}
                    ]
                return json.dumps(value)

        root, config, index, units, _ = dataset(self.base, count=2)
        config = replace(config, primary_frames=1, context_frames=2)
        manifest = run_manifest(
            index, config, units, {"model_id": "synthetic", "artifact_fingerprint": "test"}
        )
        run = self.base / "full/run"
        store = RunStore(run, manifest)
        snapshot = LabelsSnapshot(run, manifest, preflight(root, config))
        try:
            result = Runner(index, config, store, CopiesFirstFrame(), FakeLocaliser(), snapshot=snapshot).run(
                units
            )
            self.assertEqual(result["committed_frames"], 249)
            self.assertEqual(result["failed_chunks_this_invocation"], 1)
        finally:
            snapshot.close()
            store.close()
            index.close()
        labels = json.loads((self.base / "full/labels.json").read_text())
        frames = labels["sessions"]["session5"]["windows"]["window1"]["tracks"]["track_2"]["frames"]
        self.assertNotIn("0", frames)
        self.assertIn("1", frames)
        self.assertIn("249", frames)
        self.assertFalse(labels["metadata"]["indexed_annotations_complete"])

    def test_reset_archives_accepted_old_pilots_and_resume_does_not_reset(self):
        root, repo, outputs = self.base / "dataset", self.base / "repo", self.base / "outputs"
        root.mkdir()
        repo.mkdir()
        original = root / "session5/window1/global_frames/frame_000000.jpg"
        image(original)
        source_bytes = original.read_bytes()
        config = repo / "config.json"
        config.write_text("{}")
        pilot = outputs / "automatic-pilots/pilot-one"
        pilot.mkdir(parents=True)
        (pilot / "journal.jsonl").write_text("old accepted labels\n")
        (pilot.parent / "index.sqlite").write_text("reusable index")
        cache = self.base / "hf-cache"
        cache.mkdir()
        (cache / "weights").write_text("retained")
        result = prepare(outputs, root, repo, "new", config, filename="labels.py")
        archive = Path(result["archive"])
        self.assertEqual(
            (archive / "automatic-pilots/pilot-one/journal.jsonl").read_text(), "old accepted labels\n"
        )
        self.assertEqual((outputs / "full-dataset/index.sqlite").read_text(), "reusable index")
        self.assertFalse(pilot.exists())
        self.assertEqual(prepare(outputs, root, repo, "new", config, filename="labels.py"), result)
        self.assertEqual(len(list((outputs / "reset-archives").iterdir())), 1)
        with self.assertRaises(ValueError):
            prepare(outputs, root, repo, "different", config, filename="labels.py")
        self.assertEqual(original.read_bytes(), source_bytes)
        self.assertEqual((cache / "weights").read_text(), "retained")

    def test_original_labels_file_and_source_ancestor_are_protected(self):
        root, repo, outputs = self.base / "dataset", self.base / "repo", self.base / "outputs"
        root.mkdir()
        repo.mkdir()
        config = repo / "config.json"
        config.write_text("{}")
        (root / "labels.py").write_text("original = True\n")
        with self.assertRaises(ValueError):
            prepare(outputs, root, repo, "new", config, filename="labels.py")
        prepare(outputs, root, repo, "new", config, filename="labels.json")
        with self.assertRaises(ValueError):
            publish(outputs / "full-dataset/labels.json", root, "labels.py")
        with self.assertRaises(ValueError):
            publish(outputs / "full-dataset/labels.json", root, "../bad.json")
        self.assertEqual((root / "labels.py").read_text(), "original = True\n")
        with self.assertRaises(ValueError):
            prepare(self.base, root, repo, "new", config)

    def test_reset_refuses_active_writer_and_publication_is_atomic(self):
        root, repo, outputs = self.base / "dataset", self.base / "repo", self.base / "outputs"
        root.mkdir()
        repo.mkdir()
        config = repo / "config.json"
        config.write_text("{}")
        prepare(outputs, root, repo, "new", config)
        source = outputs / "full-dataset/labels.json"
        publish(source, root)
        original = (root / "labels.json").read_bytes()
        with patch("publish_labels.os.replace", side_effect=OSError("synthetic interrupted publication")):
            with self.assertRaises(OSError):
                publish(source, root)
        self.assertEqual((root / "labels.json").read_bytes(), original)
        self.assertFalse(list(root.glob(".labels*")))
        with writer_lock(outputs / "full-dataset/run/.writer.lock"):
            with self.assertRaises(RuntimeError):
                prepare(outputs, root, repo, "new", config, reset=True)
        self.assertTrue(source.exists())

    def test_explicit_bulk_flag_does_not_forge_approval(self):
        root, config, index, _, _ = dataset(self.base)
        index.close()
        args = parser().parse_args(
            [
                "run",
                "--dataset-root",
                str(root),
                "--index",
                str(self.base / "index.sqlite"),
                "--run-dir",
                str(self.base / "run"),
                "--model-dir",
                str(self.base / "model"),
                "--stage",
                "bulk",
                "--allow-unreviewed-bulk",
            ]
        )
        with patch("cue_annotation.runner.run_annotation", return_value={"complete": True}) as run:
            self.assertEqual(dispatch(args), 0)
            self.assertEqual(len(run.call_args.args[2]), 1)
        self.assertIsNone(args.pilot_approval)


FULL_DOCKER_FIXTURE = r"""
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
root = Path(os.environ["EASCCA_OUTPUTS"]) / "full-dataset"
with (root / "calls.jsonl").open("a") as stream: stream.write(json.dumps(a) + "\n")
def value(flag): return a[a.index(flag) + 1]
def host(path): return Path(os.environ["EASCCA_OUTPUTS"]) / path.removeprefix("/outputs/")
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
if a[0] in ["info", "build"]: sys.exit(0)
if a[:2] == ["image", "inspect"]:
    print("arm64" if "{{.Architecture}}" in a else "sha256:fixture"); sys.exit(0)
pos = a.index("qwen-annotation-local:spark")
c = a[pos+1]
if c == "/workspace/scripts/spark_preflight.py": write(host(value("--output")), {"passed": True})
elif c == "preflight": write(host(value("--output")), {"fixture": True})
elif c == "/workspace/scripts/full_dataset_state.py": pass
elif c == "index": host(value("--output")).write_text("fixture index")
elif c == "/workspace/scripts/pilot_suite_state.py":
    if not (root / "downloaded").exists(): sys.exit(2)
    write(host(value("--output")), {"fixture": True}); print("/hf-cache/snapshot/fixture")
elif c == "download-model": (root / "downloaded").touch()
elif c == "run":
    assert "--stage" in a and value("--stage") == "bulk" and "--allow-unreviewed-bulk" in a
    assert not any(f in a for f in ["--pilot-stage", "--max-chunks", "--max-track-windows", "--session", "--track"])
    (root / "labels.json").write_text('{"format":"eascca-indexed-labels-1.0","metadata":{"fixture":true},"sessions":{}}')
    if os.environ.get("FULL_FAIL"): sys.exit(2)
elif c == "validate": pass
else: raise RuntimeError("Unexpected operation " + c)
"""


@unittest.skipIf(os.name == "nt", "Linux Bash orchestration is exercised by Linux CI")
class FullCommandTests(unittest.TestCase):
    def test_full_scope_offline_images_reset_resume_and_failure_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            repo, root, outputs, bins = (base / n for n in ["repo", "dataset", "outputs", "bin"])
            for p in (repo / "scripts", repo / "configs", root, bins):
                p.mkdir(parents=True)
            source = Path(__file__).resolve().parents[1]
            for name in (
                "run_full_dataset.sh",
                "prepare_full_dataset.py",
                "publish_labels.py",
                "archive_empty_pilots.py",
            ):
                shutil.copy(source / "scripts" / name, repo / "scripts" / name)
            shutil.copy(source / "configs/full_dataset_7b.json", repo / "configs/full_dataset_7b.json")
            for name, body in {
                "docker": f"#!{sys.executable}\n" + FULL_DOCKER_FIXTURE,
                "uname": "#!/bin/sh\necho aarch64\n",
                "git": '#!/bin/sh\nif [ "$1" = rev-parse ]; then echo fixture; fi\nexit 0\n',
            }.items():
                path = bins / name
                path.write_text(body)
                path.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(bins) + os.pathsep + os.environ["PATH"],
                "EASCCA_DATASET": str(root),
                "EASCCA_OUTPUTS": str(outputs),
                "EASCCA_MODEL_CACHE": str(base / "cache"),
            }
            for k in ("EASCCA_CONFIG", "EASCCA_IMAGE", "EASCCA_RESET", "EASCCA_LABELS_FILENAME", "FULL_FAIL"):
                env.pop(k, None)
            old = outputs / "automatic-pilots"
            old.mkdir(parents=True)
            (old / "junk.json").write_text("old generated result")

            def execute(extra=None):
                return subprocess.run(
                    ["bash", "scripts/run_full_dataset.sh"],
                    cwd=repo,
                    env={**env, **(extra or {})},
                    capture_output=True,
                    text=True,
                    timeout=40,
                )

            result = execute({"FULL_FAIL": "1"})
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertTrue(json.loads((root / "labels.json").read_text())["metadata"]["fixture"])
            self.assertFalse(old.exists())
            result = execute()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [
                json.loads(line) for line in (outputs / "full-dataset/calls.jsonl").read_text().splitlines()
            ]
            self.assertEqual(sum("download-model" in c for c in calls), 1)
            for call in calls:
                if "bridge" in call:
                    self.assertFalse(any("dst=/dataset" in arg for arg in call))
                if "none" in call:
                    self.assertTrue(any("dst=/dataset,readonly" in arg for arg in call))
            self.assertEqual(sum(c[0] == "build" for c in calls), 1)


if __name__ == "__main__":
    unittest.main()
