# Implementation validation

On 8 October 2026, the implementation passed 39 synthetic integrity tests on Windows with Python 3.11. Tests exercise numeric indexing, exact source matching, frame ownership, recursive splitting, visibility and localisation gating, malformed model responses, token/generation limits, simulated GPU errors, committed journal digests, writer exclusion, process-crash lock release, interrupted tails, SQLite reconstruction, partial-split resume, exports and abstention-aware evaluation. Synthetic model fixtures exist only under tests and are not available as a production CLI backend.

Ruff checks, Python compilation, dependency consistency, Bash syntax and source/wheel builds also passed locally. The public CI workflow repeats core tests, lint and package construction on Windows and Linux with Python 3.10 and 3.12.

A read-only integration check used an existing preprocessed classroom window. Indexing and source validation reconciled 250 global images and 2,989 crops across 26 tracks, then verified all 3,239 source SHA256 hashes. The selected track's dry run accounted for all 250 positions in ten primary chunks without missing inputs. A geometric check located one 310 by 488 crop in its 2688 by 1520 global image using the proposed matching defaults. No semantic cue annotations were generated. This bounded check establishes neither localisation accuracy across the dataset nor behavioural annotation quality.

The ARM64 container, real Qwen checkpoint loading, CUDA memory/throughput, independent cue review and full Spark inventory remain measurements for the Spark pilot. No full-dataset annotation result or validated Spark lockfile is claimed. Follow the README and staged review protocol before a bulk run.
