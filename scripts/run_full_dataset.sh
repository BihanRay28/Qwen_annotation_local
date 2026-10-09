#!/usr/bin/env bash
# Full-session machine annotation with a single indexed JSON in the dataset root.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo="$PWD"
dataset="$(realpath "${EASCCA_DATASET:-$HOME/Desktop/preprocessed_dataset}")"
outputs="$(realpath -m "${EASCCA_OUTPUTS:-$HOME/Desktop/eascca-cue-runs}")"
cache="$(realpath -m "${EASCCA_MODEL_CACHE:-$HOME/.cache/huggingface}")"
image="${EASCCA_IMAGE:-qwen-annotation-local:spark}"
config="${EASCCA_CONFIG:-$repo/configs/full_dataset_7b.json}"
filename="${EASCCA_LABELS_FILENAME:-labels.json}"
suite="$outputs/full-dataset"
step="initial checks"
keepalive=""
publisher=""
prepared=0
cleanup() {
  code=$?
  trap - EXIT
  if [[ -n "$publisher" ]]; then kill "$publisher" 2>/dev/null || true; wait "$publisher" 2>/dev/null || true; fi
  if [[ "$prepared" == 1 && -f "$suite/labels.json" ]]; then
    python3 scripts/publish_labels.py --source "$suite/labels.json" --dataset "$dataset" --filename "$filename" || code=2
  fi
  if [[ -n "$keepalive" ]]; then kill "$keepalive" 2>/dev/null || true; fi
  exit "$code"
}
trap cleanup EXIT
trap 'code=$?; echo "Stopped during: $step (exit $code). Same command resumes accepted records. Dataset images are unchanged." >&2' ERR
[[ -d "$dataset" && -f "$config" ]]
[[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]
for path in "$outputs" "$cache"; do
  case "$path/" in "$dataset/"*|"$repo/"*) echo "Working storage must be outside dataset/repository: $path" >&2; exit 2;; esac
done
for source in "$dataset" "$repo"; do
  case "$source/" in "$cache/"*) echo "Model cache must not contain source data or repository." >&2; exit 2;; esac
done
case "$cache/" in "$outputs/automatic-pilots/"*|"$suite/"*) echo "Model cache cannot be inside generated run folders." >&2; exit 2;; esac
mkdir -p "$outputs" "$cache"
exec 8>"$(dirname "$dataset")/.$(basename "$dataset").qwen-labels.lock"
flock -n 8 || { echo "Another command is publishing labels for this dataset." >&2; exit 2; }
exec 9>"$outputs/.full-command.lock"
flock -n 9 || { echo "Another full dataset command is running." >&2; exit 2; }
revision="$(git rev-parse HEAD)"
git diff --quiet HEAD -- src scripts configs docker pyproject.toml || { echo "Commit or revert local code changes before running." >&2; exit 2; }
step="archiving old generated results"
reset=()
case "${EASCCA_RESET:-0}" in 0) :;; 1) reset=(--reset);; *) echo "EASCCA_RESET must be 0 or 1" >&2; exit 2;; esac
python3 scripts/prepare_full_dataset.py --outputs "$outputs" --dataset "$dataset" --repo "$repo" \
  --config "$config" --revision "$revision" --filename "$filename" "${reset[@]}"
prepared=1
exec > >(tee -a "$suite/full-dataset.log" 8>&- 9>&-) 2>&1
echo "Starting/resuming all sessions and all indexed tracks: $(date -Is)"
echo "Final JSON: $dataset/$filename (JSON content, even when its filename ends in .py)"
python3 scripts/publish_labels.py --source "$suite/labels.json" --dataset "$dataset" --filename "$filename"
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
  sudo -v
  docker_cmd=(sudo docker)
  (while sleep 60; do sudo -n -v || exit; done) 8>&- 9>&- </dev/null >/dev/null 2>&1 &
  keepalive=$!
fi
"${docker_cmd[@]}" info >/dev/null
step="container build"
if [[ ! -f "$suite/image-id.txt" ]]; then
  "${docker_cmd[@]}" build --platform linux/arm64 -f docker/Dockerfile.spark -t "$image" .
  "${docker_cmd[@]}" image inspect "$image" --format '{{.Id}}' > "$suite/image-id.txt"
else
  [[ "$("${docker_cmd[@]}" image inspect "$image" --format '{{.Id}}')" == "$(cat "$suite/image-id.txt")" ]]
fi
[[ "$("${docker_cmd[@]}" image inspect "$image" --format '{{.Architecture}}')" == arm64 ]]
container() {
  "${docker_cmd[@]}" run --rm --gpus all --network none --shm-size 8g \
    --user "$(id -u):$(id -g)" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --mount "type=bind,src=$dataset,dst=/dataset,readonly" \
    --mount "type=bind,src=$repo,dst=/workspace,readonly" \
    --mount "type=bind,src=$outputs,dst=/outputs" \
    --mount "type=bind,src=$cache,dst=/hf-cache" "$@"
}
qwen() { container "$image" "$@"; }
py() { container --entrypoint python "$image" "$@"; }
root=/outputs/full-dataset
cfg="$root/config.json"
step="Spark preflight"
py /workspace/scripts/spark_preflight.py --output "$root/spark-environment.json"
step="full source inventory"
if qwen preflight --dataset-root /dataset --config "$cfg" --output "$root/inventory.json"; then
  :
else
  code=$?
  [[ "$code" == 2 ]] || exit "$code"
fi
exclusions="$(py /workspace/scripts/full_dataset_state.py --report "$root/inventory.json")"
flags=()
while IFS= read -r item; do
  if [[ -n "$item" ]]; then
    echo "Unfinished source window $item: included in JSON coverage with no invented track labels."
    flags+=(--exclude-window "$item")
  fi
done <<< "$exclusions"
step="index all available sessions and tracks"
if [[ -f "$suite/index.sqlite" ]]; then
  # Stale/subset indexes are only derived files; preserve them before rebuilding.
  if ! py /workspace/scripts/full_dataset_state.py --report "$root/inventory.json" \
      --index "$root/index.sqlite" --config "$cfg"; then
    mv "$suite/index.sqlite" "$suite/index.previous-$(date +%s).sqlite"
  fi
fi
if [[ ! -f "$suite/index.sqlite" ]]; then
  qwen index --dataset-root /dataset --config "$cfg" --output "$root/index.sqlite" "${flags[@]}"
else
  qwen validate --dataset-root /dataset --index "$root/index.sqlite"
fi
step="cached model preparation"
if ! model_dir="$(py /workspace/scripts/pilot_suite_state.py model --config "$cfg" --output "$root/model.json")"; then
  [[ ! -f "$suite/model.json" ]] || { echo "Pinned model changed; restore weights or reset this run." >&2; exit 2; }
  # Online weight download cannot see the classroom images or generated labels.
  "${docker_cmd[@]}" run --rm --network bridge --user "$(id -u):$(id -g)" \
    --env HF_HUB_OFFLINE=0 --env TRANSFORMERS_OFFLINE=0 \
    --mount "type=bind,src=$cache,dst=/hf-cache" \
    --mount "type=bind,src=$suite/config.json,dst=/pilot-config.json,readonly" \
    "$image" download-model --config /pilot-config.json
  model_dir="$(py /workspace/scripts/pilot_suite_state.py model --config "$cfg" --output "$root/model.json")"
fi
step="full annotation (all sessions, all tracks, every frame)"
python3 scripts/publish_labels.py --source "$suite/labels.json" --dataset "$dataset" --filename "$filename" --watch 8>&- 9>&- &
publisher=$!
qwen run --dataset-root /dataset --index "$root/index.sqlite" --config "$cfg" \
  --run-dir "$root/run" --model-dir "$model_dir" --stage bulk --allow-unreviewed-bulk \
  --full-inventory "$root/inventory.json"
step="validate full indexed annotation coverage"
qwen validate --run-dir "$root/run" --require-complete
step="verify all source windows were available"
py /workspace/scripts/full_dataset_state.py --report "$root/inventory.json" --require-ready
echo "All indexed tracks completed. Combined JSON: $dataset/$filename"
echo "Source readiness and missing-window coverage are recorded in metadata; model accuracy remains unreviewed."
