# Sequential Spark pilots

From the cloned repository, run `bash scripts/run_all_pilots.sh`. The user has authorised sequential engineering execution of the three proposal stages. Human quality review follows these runs; the runner never approves itself or starts bulk inference.

The sequence is container build, CUDA/BF16/SDPA smoke check, actual dataset reconciliation and hashed index, frozen selection, online weights-only download, offline model pin verification, three pilot runs and complete exports, resolved environment capture, an unapproved review template and final suite validation. Stage one exercises a graceful one-chunk stop and resume. Later stages run only after the previous stage validates as complete.

An unfinished crop root blocks the normal full-dataset index. For this engineering suite, the runner explicitly records and excludes those windows in `pilot-scope.json`, leaving the source directory untouched. `preflight-full.json` retains the full inventory and original issues; `preflight.json` and the index declare the remaining subset. For example, excluding unfinished `session15/window88` from a 695-window inventory leaves a 694-window pilot index, not a complete bulk inventory. Out-of-range frames still stop the runner. Exclusions remain frozen on resume; newly unfinished windows stop it. Normal indexing without `--exclude-window SESSION:WINDOW` still rejects unfinished crop roots. Finish their original preprocessing before including them in bulk annotation.

Stage one contains one track-window. Stage five contains five tracks in five distinct classroom windows, including stage one. Stage twenty covers twenty distinct windows and up to three tracks per window, including the previous stage's windows. Repeated cases are deliberate engineering checks, not independent evaluation cases. Each stage has a separate journal and loads the model once per invocation. Each selected track processes all 250 expected positions; no inference sampling is introduced.

The selector reads indexed crop coverage and the native crop dimensions at its first, middle and last existing positions. These bounded reads are for pilot selection only. Greedy spread favours previously unrepresented sessions and diverse window positions, crop coverage and crop size. Every selected window must have all 250 global frames and at least one track with readable size probes. Unreadable probes are recorded as exclusions. The selector requires twenty eligible windows and records the selected IDs, measured proxies and fingerprints in `selection.json`. It cannot infer whether cases include occlusion, writing, reading or small-object activity. Inspect and supplement the selection under the normal review protocol if these categories are absent; report this automation as an engineering pilot until then.

Settings use the environment variables `EASCCA_DATASET`, `EASCCA_OUTPUTS`, `EASCCA_MODEL_CACHE`, `EASCCA_IMAGE`, `EASCCA_CONFIG` and `EASCCA_TRACKS_PER_WINDOW` (1 to 10, default 3). Set them before the command. Output and cache paths must be outside the dataset and repository. All suite files are under `$EASCCA_OUTPUTS/automatic-pilots`:

- `setup-and-pilots.log`: setup and inference output, including failure location.
- `selection.json`: frozen explicit IDs for all three stages.
- `pilot-scope.json`: explicit unfinished-window exclusions and their reasons.
- `model.json`, `config.json`, `code-revision.txt`, `image-id.txt`: pinned setup identity.
- `pilot-one`, `pilot-five`, `pilot-twenty`: resumable journals and audit evidence.
- `review-one`, `review-five`, `review-twenty`: full JSONL exports and local HTML/images.
- `validated-environment`: resolved package pins and GPU provenance.
- `pilot-approval.json`: an unapproved human-review template.
- `suite-summary.json`: engineering completion with `quality_reviewed=false`.

Full review-image exports can use many gigabytes and take substantial time (default stage twenty has up to 15,000 images). Check storage before starting. A successful command does not certify annotation quality. Review all stages' evidence and measured resources using `review_protocol.md` before considering bulk or a separate 3B comparison.

The script keeps source images read-only. Online download mounts only the model cache and frozen configuration file, never dataset or review images. Docker operations run through `sudo` only if direct access fails; a temporary timestamp refresher exits when the runner exits. The container still uses the calling user's UID/GID for outputs. It does not change group membership or socket permissions. One orchestration lock prevents overlapping suite writers.

Resume by rerunning the command from the same checkout and environment. Existing stages verify their run identity; existing exports verify their manifest and complete journal status. Existing model pins verify the cached artifact. Existing indices verify every source SHA256. Before any pilot manifest exists, the runner can upgrade failed setup to a new code revision and rebuild the container. After annotation has begun, changing code, configuration, container, dataset or desired track count requires a new suite location (`EASCCA_OUTPUTS`) rather than overwriting earlier evidence. Do not pull changes or rebuild the pinned image midway through an annotation run.

If using SSH, run the command in a persistent terminal such as tmux. To reconnect, use the same terminal/session or restore the environment variables before resuming. Fatal GPU errors, invalid model outputs, insufficient inventory and incompatible resume identities stop the sequence and remain visible; the runner does not repeatedly retry them without an operator diagnosing the cause.
