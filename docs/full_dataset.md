# Full-session annotation and one labels.json

The full runner implements the user's request to retire the pilot outputs and attempt every available session, classroom window, track and frame. It produces one user-facing JSON file in the original dataset root. It does not create review-image exports or stop at one/five/twenty pilot stages.

On Spark, run:

```bash
cd ~/Desktop/Qwen_annotation_local && git pull --ff-only origin main && EASCCA_DATASET="$HOME/Desktop/preprocessed_dataset" EASCCA_OUTPUTS="$HOME/Desktop/eascca-cue-runs" bash scripts/run_full_dataset.sh
```

The output is `~/Desktop/preprocessed_dataset/labels.json`. The first invocation archives the generated `automatic-pilots` suite under `~/Desktop/eascca-cue-runs/reset-archives/`. Old accepted annotations are deliberately not reused; a compatible content-hashed source index and the external downloaded weights can be reused. Source sessions, global frames, crops and preprocessing code are never moved or deleted. An existing unrecognised `labels.json` is preserved and blocks publication instead of being overwritten.

The command builds the ARM64 container, obtains Docker access with `sudo` if necessary, checks CUDA/BF16/SDPA, inventories all sessions, verifies or builds the full available-source index, resolves the cached checkpoint (downloading weights only if absent), and loads Qwen once for the full run. Online weight download mounts no dataset. All inference containers mount the dataset read-only and disable networking. A standard-library host publisher writes only the generated `labels.json` slot, using temporary files and atomic replacement. Working journals, raw audits, the sorting database and logs remain outside the dataset in `eascca-cue-runs/full-dataset`.

The full configuration requests one primary frame per model response and supplies up to two adjacent context frames on either side. Every expected frame position is still processed. This reduces the number of output records the model must emit at once; it does not establish model accuracy. No output example values or default posture labels are supplied. Invalid or copied evidence is rejected; failed visible frames remain absent from the accepted frame dictionary and their tracks remain incomplete. Missing tracked crops receive explicit unknown states with source-missing flags, preserving the existing contract.

The publication is disk-backed and streams records in numeric order instead of holding millions of labels in memory. A snapshot is written initially, after approximately fifteen minutes of accepted progress, on graceful exit and on resume. The authoritative journal commits each accepted frame before publication, so a hard power failure may leave a stale JSON snapshot but does not discard committed labels. Resume rebuilds the derived snapshot from the journal. The dataset-side publisher copies each new complete snapshot; reading the file never requires interpreting a partial JSON write.

The JSON has this lookup structure:

```python
import json
from pathlib import Path

labels = json.loads(Path.home().joinpath("Desktop/preprocessed_dataset/labels.json").read_text())
record = labels["sessions"]["session7"]["windows"]["window125"]["tracks"]["track_14846"]["frames"]["40250"]
print(record["cues"])
```

Session, window, track and source-frame IDs are explicit. Each track contains `coverage` and a `frames` dictionary. Each frame retains the sixteen explicit cue states, visibility, evidence, uncertainty/review flags, exact source paths, hashes and localisation provenance. Top-level `cue_definitions` describes the ontology. `metadata` distinguishes accepted positions, expected indexed positions, complete indexed annotation, complete source readiness, historical inventory coverage and unreviewed quality. For large finished files, prefer a streaming JSON reader to loading the whole document at once.

An unfinished crop window, such as the reported `session15/window88`, is listed in its correct session/window slot with `crop_root_not_finalised` and no invented tracks. The full source issues are also stored in metadata. Other ready windows across all sessions are processed. The command returns an incomplete status after available work if any source windows are unfinished, missing global frames or lack tracks, or if any expected model labels failed. It never calls that a complete dataset. Finish the original preprocessing before including that window; doing so changes the inventory and requires a new index/run.

Repeat the same command to resume; accepted records are not regenerated. To deliberately discard the current generated run after changing code/configuration or fixing source preprocessing, set `EASCCA_RESET=1` for that invocation:

```bash
cd ~/Desktop/Qwen_annotation_local && EASCCA_RESET=1 EASCCA_DATASET="$HOME/Desktop/preprocessed_dataset" EASCCA_OUTPUTS="$HOME/Desktop/eascca-cue-runs" bash scripts/run_full_dataset.sh
```

This archives the old full run and recognised generated labels, resets its active annotations and preserves the source data. Omit `EASCCA_RESET=1` when resuming. An old or subset index is preserved before rebuilding it. A dataset-specific publication lock and run writer locks prevent overlapping writers/resets.

The full runner explicitly uses `--stage bulk --allow-unreviewed-bulk` under the user's authorization. It does not create a fake pilot approval or claim `quality_reviewed=true`. The reviewed bulk CLI path remains available. Computational completion and correct indexing do not make model proposals educational ground truth.

There may be millions of frame positions. The JSON, raw audits and resumable checkpoints can occupy many gigabytes, and one-frame model inference can take substantially longer than the pilots. The command prints the selected scope before inference. Keep the terminal open; rerun after interruption. Monitor `~/Desktop/eascca-cue-runs/full-dataset/full-dataset.log` for progress and errors.
