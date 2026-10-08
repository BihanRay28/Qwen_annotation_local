#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
image="${EASCCA_IMAGE:-qwen-annotation-local:spark}"
dataset="${EASCCA_DATASET:-$HOME/Desktop/preprocessed_dataset}"
outputs="${EASCCA_OUTPUTS:-$HOME/Desktop/eascca-cue-runs}"
cache="${EASCCA_MODEL_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$outputs" "$cache"
if [[ "$(docker image inspect "$image" --format '{{.Architecture}}')" != "arm64" ]]; then
  echo "Expected an ARM64 image; build docker/Dockerfile.spark on the Spark." >&2
  exit 2
fi
if [[ "${1:-}" == "download-model" ]]; then
  exec docker run --rm --network bridge \
    --user "$(id -u):$(id -g)" \
    --env HF_HUB_OFFLINE=0 --env TRANSFORMERS_OFFLINE=0 \
    --mount "type=bind,src=$cache,dst=/hf-cache" \
    "$image" "$@"
fi
if [[ ! -d "$dataset" ]]; then
  echo "Dataset directory not found: $dataset" >&2
  exit 2
fi
exec docker run --rm --gpus all --network none --shm-size 8g \
  --user "$(id -u):$(id -g)" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --mount "type=bind,src=$dataset,dst=/dataset,readonly" \
  --mount "type=bind,src=$PWD,dst=/workspace,readonly" \
  --mount "type=bind,src=$outputs,dst=/outputs" \
  --mount "type=bind,src=$cache,dst=/hf-cache" \
  "$image" "$@"

