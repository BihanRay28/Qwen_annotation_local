# Implementation coverage

This repository implements the local annotation proposal and the code plan without changing the report, existing preprocessing or independent Pass 2 annotations.

| Plan requirement | Implementation |
| --- | --- |
| Pathlib dataset root and numeric regex IDs | `dataset.py` and CLI dataset-root option |
| Exact absolute frame pairing and time f/25 | Content-hashed SQLite index and generated per-frame source records |
| All 250 positions and 25-frame calls | `chunking.py`, overlap and recursive primary-range splitting |
| Batch size 1 DataLoader | `QwenBackend.iter_tasks` with zero workers and one model process |
| Explicit crop/global multimodal frame labels | `prompts.py` |
| Qwen AutoProcessor and local model tensors | `qwen_backend.py`, BF16 SDPA and measured encoded token length |
| Target highlight and localisation | `localisation.py`, native refinement and distinct-match rejection |
| Sixteen simultaneous visible cues and unknowns | `ontology.py`, strict Pydantic schema and host visibility gating |
| Source hashes, evidence and review flags | Expanded host-controlled `FrameRecord` |
| Crash recovery and duplicate rejection | Single-writer journal, fsync, replayable SQLite and run fingerprints |
| Numeric JSONL exports and supported intervals | `export.py`, with optional local review images |
| Independent cue evaluation | `evaluation.py`, with explicit abstention and reference denominators |
| Spark deployment and offline execution | ARM64 Dockerfile, shell helper and GPU preflight |
| Environment freeze after pilot | `scripts/freeze_environment.py` |
| Synthetic integrity checks without imagery | `tests`, CPU CI on Windows and Linux |

The GitHub repository contains code, configuration, tests and documentation only. It has no classroom imagery, existing annotation files, model weights, private run outputs or report binaries. Git and Docker exclusions enforce this boundary.

Core integrity and CPU localisation are tested locally. Full Qwen loading, ARM64 container build, actual GPU memory/throughput and cue quality remain Spark pilot measurements. The implementation supplies the commands and acceptance checks; it does not claim those future results or estimate a full-run completion date.
