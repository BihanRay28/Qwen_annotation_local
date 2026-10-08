# Dataset contract

Set the root to the actual `preprocessed_dataset` folder, not its parent. Expected structure:

```text
sessionN/windowM/global_frames/frame_XXXXXX.jpg
sessionN/windowM/cropped_frames_per_student/track_ID/frame_XXXXXX.jpg
```

Numeric directory sorting uses full regular-expression matches. `.jpg` and `.jpeg` frame filenames are recognised case-insensitively, but duplicate numeric IDs or frames are errors. Symbolic input directories/files are rejected. A global image and crop are paired only when the absolute source frame indices match. Nominal time is `source_frame / 25` seconds. Nearest-frame substitution is not used.

Each numbered window owns 250 expected positions. The first expected frame is computed from the configured retained session start plus `(window_id - 1) * 250`; it is not guessed from the earliest surviving image. The proposal defaults are session starts 5=0, 7=9225, 8=0, 10=8725, 11=0, 14=85000, 15=0. Check them against the actual source filenames. Unknown sessions require an explicit `session_start_frames` mapping.

Preflight distinguishes the historical seven-session, 695-window scope from a subset or changed inventory. It reports missing globals, crop availability, absent windows, empty track sets, unfinished crop roots and out-of-range filenames. An unfinished `cropped_frames_per_student.__partial` folder blocks indexing; repair or reconcile preprocessing on the Spark first. Missing crops do not prove absence, inattention or extraction failure. All detected tracks remain eligible for later review, including possible non-students and mixed identities.

The SQLite index stores existing relative source paths, SHA256, size and modification time, plus deterministic windows and track IDs. All expected per-track frame positions are generated from that index, including explicit missingness. Both index construction and runtime check source integrity. Runtime first rechecks the inventory and then hashes every image supplied to the model. `validate --index ... --dataset-root ...` rehashes the complete indexed source set, even inputs not used in a particular pilot.

Index content fingerprints do not depend on absolute machine paths. Run manifests retain the dataset root for local review/export. Build the index on the Spark against its actual source tree; do not reuse a Windows inventory as proof of Spark completeness. New or modified data requires a new index and a new run.

Image resolution is preserved until the Qwen processor applies per-image pixel bounds with aspect ratio. A universal 224 by 224 input crop size is not assumed. Output paths are checked to remain outside the dataset, and the Spark helper mounts it read-only.

