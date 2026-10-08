"""Create an UNAPPROVED template from a completed pilot, for independent review."""

import argparse
import json
from pathlib import Path

from cue_annotation.persistence import inspect_run
from cue_annotation.util import atomic_json, outside

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
run = args.run_dir.expanduser().resolve()
if not inspect_run(run)["complete"]:
    raise SystemExit("Pilot is incomplete; finish/recover it before creating an approval template")
manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
approval = {
    k: manifest[k]
    for k in (
        "dataset_fingerprint",
        "config_fingerprint",
        "ontology_fingerprint",
        "prompt_fingerprint",
        "schema_fingerprint",
    )
}
approval.update(
    model_artifact_fingerprint=manifest["model"]["artifact_fingerprint"],
    quality_reviewed=False,
    resources_feasible=False,
    approved_by="",
    review_notes="",
)
output = outside(args.output, Path(manifest["dataset_root"]))
if output.exists():
    raise SystemExit("Output exists; approval documents are not overwritten")
atomic_json(output, approval)
print(
    "Created an unapproved template. Review cue quality, failure cases and measured resources before approving."
)
