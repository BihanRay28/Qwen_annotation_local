#!/usr/bin/env bash
# One command: setup -> one full track -> five tracks -> twenty classroom windows.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo="$PWD"
dataset="$(realpath "${EASCCA_DATASET:-$HOME/Desktop/preprocessed_dataset}")"
outputs="$(realpath -m "${EASCCA_OUTPUTS:-$HOME/Desktop/eascca-cue-runs}")"
cache="$(realpath -m "${EASCCA_MODEL_CACHE:-$HOME/.cache/huggingface}")"
image="${EASCCA_IMAGE:-qwen-annotation-local:spark}"
config="${EASCCA_CONFIG:-$repo/configs/all_frames_7b.json}"
tracks="${EASCCA_TRACKS_PER_WINDOW:-3}"
suite="$outputs/automatic-pilots"
step="initial checks"
keepalive=""
cleanup() { if [[ -n "$keepalive" ]]; then kill "$keepalive" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'code=$?; echo "Stopped during: $step (exit $code). Fix the reported error, then rerun the same command to resume." >&2' ERR
[[ -d "$dataset" && -f "$config" ]]
[[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]
for path in "$outputs" "$cache"; do
  case "$path/" in "$dataset/"*|"$repo/"*) echo "Storage must be outside dataset and repository: $path" >&2; exit 2;; esac
done
mkdir -p "$suite" "$cache"
exec 9>"$suite/.suite.lock"
flock -n 9 || { echo "Another pilot suite is already running." >&2; exit 2; }
exec > >(tee -a "$suite/setup-and-pilots.log" 9>&-) 2>&1
echo "Starting/resuming all engineering pilots: $(date -Is)"
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
  echo "Docker requires administrator access; enter your login password if prompted."
  sudo -v
  docker_cmd=(sudo docker)
  # Refresh this session's sudo timestamp only while this runner is alive.
  (while sleep 60; do sudo -n -v || exit; done) 9>&- </dev/null >/dev/null 2>&1 &
  keepalive=$!
fi
"${docker_cmd[@]}" info >/dev/null
if [[ -f "$suite/config.json" ]]; then
  cmp -s "$config" "$suite/config.json" || { echo "Configuration changed. Choose a new EASCCA_OUTPUTS directory." >&2; exit 2; }
else
  cp "$config" "$suite/config.json.tmp"
  mv "$suite/config.json.tmp" "$suite/config.json"
fi
revision="$(git rev-parse HEAD)"
git diff --quiet HEAD -- src scripts configs docker pyproject.toml || { echo "Commit or revert code changes before this reproducible pilot." >&2; exit 2; }
if [[ -f "$suite/code-revision.txt" ]]; then
  [[ "$(cat "$suite/code-revision.txt")" == "$revision" ]] || { echo "Code changed. Choose a new EASCCA_OUTPUTS directory." >&2; exit 2; }
else
  printf '%s\n' "$revision" > "$suite/code-revision.txt.tmp"
  mv "$suite/code-revision.txt.tmp" "$suite/code-revision.txt"
fi
step="container build"
if [[ -f "$suite/image-id.txt" ]]; then
  [[ "$("${docker_cmd[@]}" image inspect "$image" --format '{{.Id}}')" == "$(cat "$suite/image-id.txt")" ]] || { echo "Saved container image changed or disappeared. Restore it or use a new EASCCA_OUTPUTS directory." >&2; exit 2; }
else
  "${docker_cmd[@]}" build --platform linux/arm64 -f docker/Dockerfile.spark -t "$image" .
  "${docker_cmd[@]}" image inspect "$image" --format '{{.Id}}' > "$suite/image-id.txt.tmp"
  mv "$suite/image-id.txt.tmp" "$suite/image-id.txt"
fi
[[ "$("${docker_cmd[@]}" image inspect "$image" --format '{{.Architecture}}')" == "arm64" ]]
"${docker_cmd[@]}" image inspect "$image" > "$suite/annotation-image.json"
qwen() {
  "${docker_cmd[@]}" run --rm --gpus all --network none --shm-size 8g \
    --user "$(id -u):$(id -g)" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --mount "type=bind,src=$dataset,dst=/dataset,readonly" \
    --mount "type=bind,src=$repo,dst=/workspace,readonly" \
    --mount "type=bind,src=$outputs,dst=/outputs" \
    --mount "type=bind,src=$cache,dst=/hf-cache" "$image" "$@"
}
py() {
  "${docker_cmd[@]}" run --rm --gpus all --network none --shm-size 8g \
    --entrypoint python --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$dataset,dst=/dataset,readonly" \
    --mount "type=bind,src=$repo,dst=/workspace,readonly" \
    --mount "type=bind,src=$outputs,dst=/outputs" \
    --mount "type=bind,src=$cache,dst=/hf-cache" "$image" "$@"
}
root=/outputs/automatic-pilots
cfg="$root/config.json"
step="Spark CUDA/BF16/SDPA preflight"
py /workspace/scripts/spark_preflight.py --output "$root/spark-environment.json"
step="dataset reconciliation and indexing"
qwen preflight --dataset-root /dataset --config "$cfg" --output "$root/preflight.json"
if [[ ! -f "$suite/index.sqlite" ]]; then
  qwen index --dataset-root /dataset --config "$cfg" --output "$root/index.sqlite"
else
  qwen validate --dataset-root /dataset --index "$root/index.sqlite"
fi
step="freezing pilot selections"
qwen plan-pilots --dataset-root /dataset --config "$cfg" --index "$root/index.sqlite" \
  --output "$root/selection.json" --tracks-per-window "$tracks"
step="model download and immutable checkpoint pin"
if [[ ! -f "$suite/model.json" ]]; then
  # Online download mounts the cache and frozen configuration, never classroom images.
  "${docker_cmd[@]}" run --rm --network bridge --user "$(id -u):$(id -g)" \
    --env HF_HUB_OFFLINE=0 --env TRANSFORMERS_OFFLINE=0 \
    --mount "type=bind,src=$cache,dst=/hf-cache" \
    --mount "type=bind,src=$suite/config.json,dst=/pilot-config.json,readonly" \
    "$image" download-model --config /pilot-config.json
fi
model_dir="$(py /workspace/scripts/pilot_suite_state.py model --config "$cfg" --output "$root/model.json")"
for stage in one five twenty; do
  step="pilot-$stage"
  echo "========== $step =========="
  selection=(--dataset-root /dataset --index "$root/index.sqlite" --config "$cfg" \
    --units-file "$root/selection.json" --pilot-stage "$stage")
  qwen run "${selection[@]}" --run-dir "$root/plan-$stage" --dry-run
  if [[ "$stage" == one ]]; then
    qwen run "${selection[@]}" --run-dir "$root/pilot-$stage" --model-dir "$model_dir" --max-chunks 1
    qwen status --run-dir "$root/pilot-$stage"
  fi
  qwen run "${selection[@]}" --run-dir "$root/pilot-$stage" --model-dir "$model_dir"
  qwen validate --run-dir "$root/pilot-$stage" --require-complete
  step="export pilot-$stage"
  if [[ ! -d "$suite/review-$stage" ]]; then
    # Includes every output position, not only the first track's review prefix.
    qwen export --run-dir "$root/pilot-$stage" --dataset-root /dataset \
      --output "$root/review-$stage" --review-images 150000
  else
    py /workspace/scripts/pilot_suite_state.py check-export --run-dir "$root/pilot-$stage" --output "$root/review-$stage"
  fi
done
step="environment capture"
if [[ ! -d "$suite/validated-environment" ]]; then
  py /workspace/scripts/freeze_environment.py --pilot-run "$root/pilot-twenty" --output-dir "$root/validated-environment"
fi
step="unapproved review template"
if [[ ! -f "$suite/pilot-approval.json" ]]; then
  py /workspace/scripts/create_approval_template.py --run-dir "$root/pilot-twenty" --output "$root/pilot-approval.json"
fi
step="final suite validation"
py /workspace/scripts/pilot_suite_state.py finish --root "$root"
echo "All three engineering pilots completed. Results: $suite"
echo "Review review-one/review.html, review-five/review.html and review-twenty/review.html."
echo "Cue quality and behavioural diversity still require human review. No bulk run was started."
