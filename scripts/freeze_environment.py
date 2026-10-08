"""Record resolved pins only after a complete Spark model pilot."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from cue_annotation.persistence import inspect_run
from cue_annotation.util import atomic_json, outside

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pilot-run", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
args = parser.parse_args()
run = args.pilot_run.expanduser().resolve()
status = inspect_run(run)
manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
if not status["complete"] or platform.machine().lower() not in {"aarch64", "arm64"}:
    raise SystemExit("Require a complete pilot on ARM64 Spark before publishing validated pins")
import torch  # noqa: E402 -- GPU import follows the explicit completed-pilot checks.

if not torch.cuda.is_available():
    raise SystemExit("CUDA environment required")
output = outside(args.output_dir, Path(manifest["dataset_root"]))
if output.exists():
    raise SystemExit("Output directory exists; use a new path")
output.mkdir(parents=True)
pins = subprocess.run(
    [sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True
)
(output / "requirements.spark.lock.txt").write_text(pins.stdout, encoding="utf-8")
atomic_json(
    output / "validated_environment.json",
    {
        "pilot_run_fingerprint": status["run_fingerprint"],
        "model": manifest["model"],
        "packages": manifest["environment"],
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "note": "Engineering smoke test completed; cue quality requires independent review",
    },
)
print(output)
