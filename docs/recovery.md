# Durable annotation and recovery

`journal.jsonl` is authoritative. Each newline-terminated `chunk_commit` event contains all validated frame records for that primary range, their keys, a digest and the run fingerprint. It is flushed and fsynced before the derived `state.sqlite` database is updated. The operating-system writer lock is released automatically if a process crashes; the presence of its lock file does not mean the lock remains held.

On resume, the manifest must match the dataset, selected units, model artifact, code, prompt, ontology, schema, configuration and package environment. The runner replays committed journal events and rebuilds SQLite. Accepted primary frames are skipped. If only one recursively split half was committed before interruption, only the remaining range is recomputed. Conflicting duplicates or overlapping primary ownership are errors.

An unterminated final journal line is uncommitted. Resume preserves it in an `.interrupted-<byte offset>` audit file before truncating it under the writer lock. A malformed newline-terminated event, changed digest or corrupted middle event is an error, not an excuse to discard records. Preserve the run directory and diagnose it rather than manually deleting committed lines.

`status --run-dir ...` and `validate --run-dir ...` read the journal without loading Qwen. Status can be checked while inference runs; a very brief unfinished final write may require retrying the read after the active chunk completes. `validate --require-complete` returns a nonzero exit status until every expected frame in the declared selection is committed. Source-missing positions can have valid unknown records; unreadable images or unresolved inference failures keep their chunk incomplete.

`--max-chunks N` is a graceful interruption exercise and is not part of the run fingerprint. Resume with the same selection and without that limit to continue. A fatal CUDA process error exits; restart using the same command. A new source inventory, protocol or software environment requires a new run directory.

Exports are separate directories. `frames.jsonl` is ordered numerically by session/window/track/frame; `track_windows.jsonl` provides explicit cue counts and contiguous supported intervals. Missing or unknown observations break intervals. No minimum-duration filter or engagement aggregation is applied. Review images are optional local files and must not be committed to Git.

