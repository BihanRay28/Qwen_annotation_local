"""Exercise the actual Bash orchestration with a synthetic Docker command boundary."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DOCKER_FIXTURE = r"""
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
root = Path(os.environ["EASCCA_OUTPUTS"]) / "automatic-pilots"
root.mkdir(parents=True, exist_ok=True)
with (root / "fixture-calls.jsonl").open("a") as stream:
    stream.write(json.dumps(a) + "\n")
def value(flag):
    return a[a.index(flag) + 1]
def host(path):
    return Path(os.environ["EASCCA_OUTPUTS"]) / path.removeprefix("/outputs/")
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
if a[0] == "info" or a[0] == "build":
    sys.exit(0)
if a[:2] == ["image", "inspect"]:
    print("arm64" if "{{.Architecture}}" in a else "sha256:fixture" if "{{.Id}}" in a else "[]")
    sys.exit(0)
if a[0] != "run":
    raise RuntimeError("Unexpected Docker operation")
image_pos = a.index("qwen-annotation-local:spark")
command = a[image_pos + 1]
if command == "/workspace/scripts/spark_preflight.py":
    write(host(value("--output")), {"passed": True})
elif command == "/workspace/scripts/pilot_suite_state.py":
    mode = a[image_pos + 2]
    if mode == "model":
        write(host(value("--output")), {"fixture": True})
        print("/hf-cache/snapshots/fixture")
    elif mode == "check-export":
        assert (host(value("--output")) / "review.html").is_file()
    elif mode == "scope":
        exclusions = [{"session": 15, "window": 88, "reason": "crop_root_not_finalised"}] if os.environ.get("UNFINISHED_WINDOW") else []
        write(host(value("--output")), {"exclusions": exclusions})
        if exclusions:
            print("15:88")
    elif mode == "finish":
        assert all((root / ("pilot-" + s) / "complete.json").exists() for s in ["one", "five", "twenty"])
        write(root / "suite-summary.json", {"engineering_complete": True, "quality_reviewed": False})
    else:
        raise RuntimeError("Unexpected state operation")
elif command == "/workspace/scripts/freeze_environment.py":
    write(host(value("--output-dir")) / "validated_environment.json", {"fixture": True})
elif command == "/workspace/scripts/create_approval_template.py":
    write(host(value("--output")), {"quality_reviewed": False, "resources_feasible": False})
elif command == "preflight":
    write(host(value("--output")), {"windows": 20})
    if os.environ.get("UNFINISHED_WINDOW") and "preflight-full.json" in value("--output"):
        sys.exit(2)
elif command == "index":
    host(value("--output")).write_text("synthetic index boundary")
elif command == "plan-pilots":
    write(host(value("--output")), {"fixture": True})
elif command == "download-model":
    pass
elif command == "run":
    stage = value("--pilot-stage")
    if "--dry-run" not in a and os.environ.get("FAIL_STAGE") == stage:
        print("Synthetic stage failure", file=sys.stderr)
        sys.exit(2)
    run = host(value("--run-dir"))
    run.mkdir(parents=True, exist_ok=True)
    if "--dry-run" not in a:
        write(run / "manifest.json", {"fixture": True})
    if "--dry-run" not in a and "--max-chunks" not in a:
        write(run / "complete.json", {"complete": True})
elif command == "validate":
    if "--require-complete" in a:
        assert (host(value("--run-dir")) / "complete.json").is_file()
elif command == "status":
    pass
elif command == "export":
    write(host(value("--output")) / "review.html", {"fixture": True})
else:
    raise RuntimeError("Unexpected command " + command)
"""


@unittest.skipIf(os.name == "nt", "Linux Bash orchestration is exercised by Linux CI")
class SequentialRunnerTests(unittest.TestCase):
    def prepare(self, base):
        repo = base / "repo"
        (repo / "scripts").mkdir(parents=True)
        (repo / "configs").mkdir()
        source = Path(__file__).resolve().parents[1]
        shutil.copy(source / "scripts/run_all_pilots.sh", repo / "scripts/run_all_pilots.sh")
        shutil.copy(source / "configs/all_frames_7b.json", repo / "configs/all_frames_7b.json")
        dataset = base / "dataset"
        dataset.mkdir()
        binaries = base / "bin"
        binaries.mkdir()
        for name, body in {
            "docker": f"#!{sys.executable}\n" + DOCKER_FIXTURE,
            "uname": "#!/bin/sh\necho aarch64\n",
            "git": '#!/bin/sh\nif [ "$1" = rev-parse ]; then echo "${FIXTURE_REVISION:-fixture-revision}"; fi\nexit 0\n',
        }.items():
            path = binaries / name
            path.write_text(body)
            path.chmod(0o755)
        env = {
            **os.environ,
            "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
            "EASCCA_DATASET": str(dataset),
            "EASCCA_OUTPUTS": str(base / "outputs"),
            "EASCCA_MODEL_CACHE": str(base / "cache"),
            "EASCCA_IMAGE": "qwen-annotation-local:spark",
        }
        env.pop("EASCCA_CONFIG", None)
        env.pop("EASCCA_TRACKS_PER_WINDOW", None)
        env.pop("FAIL_STAGE", None)
        env.pop("UNFINISHED_WINDOW", None)
        env.pop("FIXTURE_REVISION", None)
        return repo, env, base / "outputs/automatic-pilots"

    def execute(self, repo, env):
        return subprocess.run(
            ["bash", "scripts/run_all_pilots.sh"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=40,
        )

    def test_all_stages_are_sequential_and_repeat_is_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, env, root = self.prepare(Path(temporary))
            result = self.execute(repo, env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in (root / "fixture-calls.jsonl").read_text().splitlines()]
            full_runs = [
                a[a.index("--pilot-stage") + 1]
                for a in calls
                if "--pilot-stage" in a and "--dry-run" not in a and "--max-chunks" not in a
            ]
            self.assertEqual(full_runs, ["one", "five", "twenty"])
            online = [a for a in calls if "bridge" in a]
            self.assertEqual(len(online), 1)
            self.assertFalse(
                any(
                    "dst=/dataset" in arg or "dst=/outputs" in arg or "dst=/suite" in arg for arg in online[0]
                )
            )
            self.assertTrue(json.loads((root / "suite-summary.json").read_text())["engineering_complete"])
            self.assertFalse(json.loads((root / "pilot-approval.json").read_text())["quality_reviewed"])
            initial_count = len(calls)
            result = self.execute(repo, env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            repeated = [json.loads(line) for line in (root / "fixture-calls.jsonl").read_text().splitlines()][
                initial_count:
            ]
            self.assertFalse(
                any(
                    a[0] == "build" or "download-model" in a or "index" in a or "export" in a
                    for a in repeated
                )
            )

    def test_failure_stops_later_stages_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, env, root = self.prepare(Path(temporary))
            result = self.execute(repo, {**env, "FAIL_STAGE": "five"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Stopped during: pilot-five", result.stdout + result.stderr)
            self.assertFalse((root / "pilot-twenty").exists())
            self.assertFalse((root / "suite-summary.json").exists())
            result = self.execute(repo, env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / "suite-summary.json").is_file())

    def test_upgrade_before_annotation_and_explicit_partial_window_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, env, root = self.prepare(Path(temporary))
            root.mkdir(parents=True)
            (root / "code-revision.txt").write_text("old-revision\n")
            (root / "image-id.txt").write_text("sha256:old-image\n")
            result = self.execute(repo, {**env, "UNFINISHED_WINDOW": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Upgrading setup", result.stdout)
            self.assertIn("Pilot index exclusion: 15:88", result.stdout)
            calls = [json.loads(line) for line in (root / "fixture-calls.jsonl").read_text().splitlines()]
            index_calls = [call for call in calls if "index" in call]
            self.assertEqual(index_calls[0][-2:], ["--exclude-window", "15:88"])
            result = self.execute(repo, {**env, "FIXTURE_REVISION": "changed-after-pilots"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Code changed after annotation began", result.stdout + result.stderr)
