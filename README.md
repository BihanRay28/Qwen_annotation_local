# Qwen annotation local

This package implements the EASCCA local Qwen cue annotation proposal and its Markdown code plan. It pairs each tracked crop with the global classroom image at the **same absolute source frame**, then uses Qwen2.5-VL-7B-Instruct locally on DGX Spark. Every complete ten-second window has 250 expected positions at 25 FPS. No temporal sampling is used for production annotation.

The outputs describe observable cues with evidence, visibility, uncertainty and review flags. They do not assign engagement, emotion, attention or cognitive labels. Tracking IDs remain session-local and every detected track is retained for human eligibility review. A later 3B comparison can use the same interface and reviewed cases.

Source images and preprocessing are read-only. The full runner publishes only its generated `labels.json` into the dataset root. Images, weights, indices, journals, exports and caches stay outside this repository. The independent Pass 2 workflow and existing preprocessing are not modified.

## Full dataset in one command

Run this in the Spark terminal:

```bash
cd ~/Desktop/Qwen_annotation_local && git pull --ff-only origin main && EASCCA_DATASET="$HOME/Desktop/preprocessed_dataset" EASCCA_OUTPUTS="$HOME/Desktop/eascca-cue-runs" bash scripts/run_full_dataset.sh
```

This archives old generated pilot results on its first invocation, retains compatible indexing and cached weights, sets up the container, and attempts every available session, window, track and frame. The result is **`~/Desktop/preprocessed_dataset/labels.json`**, indexed by session → window → track → absolute source frame. One primary frame per response reduces multi-frame copying. All sixteen cue states and their evidence/provenance are retained. Repeating the command resumes; deliberate resets use `EASCCA_RESET=1` and preserve the previous run in an external archive.

The source images remain mounted read-only. The JSON is updated atomically, periodically and on exit; journals and raw audits stay outside the dataset. Unfinished source windows remain explicitly unlabelled in the JSON, and failed model outputs do not become fabricated labels. If any such gaps remain, the command reports incomplete coverage after available work. The full run can cover millions of positions and take substantially longer than the pilots. Machine accuracy still requires review. See [full-dataset execution](docs/full_dataset.md).

## Spark quick start

The following is the optional staged pilot/review workflow. Use the full-dataset command above for all-session execution.

For a single sequential setup and all three engineering pilots, after cloning run:

```bash
cd ~/Desktop/Qwen_annotation_local
git pull --ff-only
bash scripts/run_all_pilots.sh
```

The runner builds the container, uses `sudo` if Docker requires it, checks the GPU, indexes the dataset, freezes explicit selections, downloads and pins the 7B checkpoint, then runs one track-window (including save/resume), five tracks in distinct windows, and up to three tracks in each of twenty classroom windows. Every selected track processes all 250 positions. It validates each stage, exports every position for visual review, captures the environment and creates an **unapproved** review template. It stops on errors; repeating the command resumes the same selections and checkpoint. See [the sequential runner](docs/sequential_pilots.md).

Automatic selection spreads sessions, window positions, crop sizes and missing-frame rates. It does not establish behavioural diversity or cue accuracy; the operator must confirm the proposal's diversity categories and review quality. It requires twenty usable classroom windows, and does not silently substitute a smaller pilot. Default paths are the same as the helper below. All results go to `~/Desktop/eascca-cue-runs/automatic-pilots`.

Unfinished crop windows are explicitly excluded from this engineering pilot index and recorded in `pilot-scope.json`; the source dataset stays untouched. Full inventory and scoped pilot reports are both retained. Out-of-range frames still stop processing, and unfinished windows must be repaired before including them in bulk annotation.

If an older protocol failed with zero accepted records, pulling an update and rerunning preserves those failed run folders under `failed-empty-pilots` and restarts with the revised protocol while retaining the index, selections and cached weights. Any accepted records prevent this upgrade; use a new output location for a new protocol. Known example/placeholder evidence is rejected rather than converted into annotation labels.

Clone onto the Spark and build the proposed ARM64 GPU container:

```bash
cd ~/Desktop
git clone https://github.com/BihanRay28/Qwen_annotation_local.git
cd Qwen_annotation_local
docker build --platform linux/arm64 -f docker/Dockerfile.spark -t qwen-annotation-local:spark .
```

The base is NVIDIA PyTorch `25.09-py3`, a proposal starting point. The Dockerfile preserves NVIDIA's PyTorch/CUDA stack. This repository's CPU checks do not establish Spark compatibility or annotation accuracy. Complete the actual Spark preflight and one-track pilot before freezing versions or scheduling a bulk run. See [Spark execution](docs/spark_execution.md) for the GPU check and environment capture.

Download the open model weights once. This is the only helper command that enables networking; it does not mount classroom data:

```bash
bash scripts/spark.sh download-model --config configs/all_frames_7b.json
```

The helper maps `~/Desktop/preprocessed_dataset` to `/dataset` read-only, run storage to `/outputs`, and the model cache to `/hf-cache`. Other commands run with networking disabled. Set `EASCCA_DATASET`, `EASCCA_OUTPUTS` or `EASCCA_MODEL_CACHE` if your locations differ.

Reconcile the actual dataset and build its content-hashed index. The index may take substantial time because it reads every image to calculate SHA256; progress is printed after each window. A subset is reported explicitly instead of silently described as the historical 695-window inventory.

```bash
bash scripts/spark.sh preflight --dataset-root /dataset --output /outputs/preflight.json
bash scripts/spark.sh index --dataset-root /dataset --output /outputs/index.sqlite
bash scripts/spark.sh list-units --dataset-root /dataset --index /outputs/index.sqlite --limit 20
```

The first pilot processes one full track-window in numeric order. Inspect `list-units` before choosing explicit `--session`, `--window` and `--track` IDs. A dry run creates an input plan and no annotations:

```bash
bash scripts/spark.sh run --dataset-root /dataset --index /outputs/index.sqlite \
  --run-dir /outputs/plan-one --max-track-windows 1 --dry-run

bash scripts/spark.sh run --dataset-root /dataset --index /outputs/index.sqlite \
  --run-dir /outputs/pilot-one --max-track-windows 1

bash scripts/spark.sh validate --run-dir /outputs/pilot-one --require-complete
bash scripts/spark.sh export --run-dir /outputs/pilot-one --dataset-root /dataset \
  --output /outputs/pilot-one-review --review-images 25
```

Open `~/Desktop/eascca-cue-runs/pilot-one-review/review.html` locally. These images are a bounded inspection prefix, not an independently selected evaluation sample. Review all 250 output positions, target localisation, JSON evidence, and the `audit` and `metrics.jsonl` files. A valid JSON response alone does not establish cue accuracy.

To stop and resume, rerun the **same command with the same run directory, index, selected units, model and configuration**. `--max-chunks 1` stops after one newly committed chunk for an interruption exercise; omit it on the next invocation to continue. Changing the prompt, ontology, code, environment, model, dataset or configuration requires a new run directory. Never run two writers against one directory.

Expand to five deliberately diverse track-windows, then approximately twenty classroom windows with selected tracks. Use the selection flags and separate run directories. Do not mistake the first five numeric units for a diverse sample. After independent review, create and complete a pilot approval document as described in [pilot review](docs/review_protocol.md). Full processing requires an explicit bulk stage:

```bash
bash scripts/spark.sh run --dataset-root /dataset --index /outputs/index.sqlite \
  --run-dir /outputs/bulk-7b --stage bulk --pilot-approval /outputs/pilot-approval.json
```

## Local core checks

Python 3.10 or later is sufficient for indexing, validation, exports and synthetic tests. No GPU or model weights are required:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]' 'opencv-python-headless>=4.10,<5'
python -m unittest discover -s tests -v
ruff check src tests scripts
eascca-cues --help
```

For Qwen inference install `.[vlm]` **inside a functioning NVIDIA Spark GPU environment**. PyTorch is intentionally supplied by that environment rather than replaced by this package. The VLM dependency ranges are provisional until the Spark pilot; `scripts/freeze_environment.py` records resolved versions after an engineering pilot.

## Contracts and implementation

- [Dataset contract](docs/dataset_contract.md): absolute frame mapping, missing inputs and source hashes.
- [Cue protocol](docs/annotation_protocol.md): sixteen cues, three states and visibility gating.
- [Recovery](docs/recovery.md): journal authority, partial writes, resume identity and conflicts.
- [Spark execution](docs/spark_execution.md): offline containers, GPU checks and version capture.
- [Review and model comparison](docs/review_protocol.md): independent review, release gate and evaluation metrics.
- [Implementation coverage](docs/implementation.md): mapping from the report and code plan to modules.
- [Validation performed](docs/validation.md): local checks and Spark work still to measure.

The CLI includes `preflight`, `index`, `download-model`, `run`, `status`, `validate`, `export`, `list-units` and `evaluate`. Outputs never become educational ground truth solely because the run completes.
