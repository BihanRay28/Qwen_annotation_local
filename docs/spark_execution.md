# DGX Spark execution

The user runs deployment on their Spark. Build with `docker build --platform linux/arm64 -f docker/Dockerfile.spark -t qwen-annotation-local:spark .`. NVIDIA's `25.09-py3` base follows the proposal and is configurable with `--build-arg BASE_IMAGE=...`. Architecture, driver, CUDA, BF16, SDPA, processor token length and one full model track-window still require verification on the actual device. Do not treat a successful container build as a model or annotation test.

The VLM extra supplies Transformers, Accelerate, Qwen vision utilities, Hugging Face Hub and headless OpenCV. NVIDIA's preinstalled GPU PyTorch supplies CUDA kernels; installing a generic PyTorch wheel over it can break Spark support. The base image's dependency constraints are retained. Resolve any package conflict deliberately, rebuild, rerun `pip check` and record the resulting environment. No Spark-validated lockfile is claimed in this initial implementation.

After building, check actual GPU execution:

```bash
mkdir -p ~/Desktop/eascca-cue-runs
docker run --rm --gpus all --network none --entrypoint python \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$HOME/Desktop/eascca-cue-runs,dst=/outputs" \
  qwen-annotation-local:spark /workspace/scripts/spark_preflight.py \
  --output /outputs/spark-environment.json
docker image inspect qwen-annotation-local:spark --format '{{json .RepoDigests}}'
```

Retain the built image ID/digest and base digest alongside that environment output. The check exercises BF16 scaled dot product attention on CUDA, but does not load Qwen. Then follow the one-track model pilot in the README.

`download-model` resolves the requested Hugging Face revision to an immutable snapshot and downloads open model files only. It calculates an artifact fingerprint and prints the cached snapshot path. Hashing model weights takes time. Inference loads the cache with `local_files_only=True` and `trust_remote_code=False`. The runner records the actual checkpoint revision, artifact fingerprint and installed package versions. It refuses context budgets above the checkpoint's configured limit.

Initial defaults are batch size 1, zero DataLoader workers, BF16, SDPA, greedy decoding, 57,344 input tokens and 6,144 generated tokens. Each global image permits up to `1024 * 28 * 28` pixels and each crop up to `512 * 28 * 28`; the minimum is `4 * 28 * 28`. The processor's actual input tensor length governs splitting. A 35-pair interior call can approach roughly 53,760 visual tokens before text, so split frequency must be measured. These are starting budgets, not guaranteed memory or throughput results.

The shell helper uses a single GPU inference process, a bounded global-frame cache, read-only data/code mounts and local writable output/cache mounts. It passes the current user's UID/GID so run files remain accessible. It disables networking for preflight, indexing, inference, validation and export. It does not start a server or expose ports. For long runs use a persistent terminal such as tmux and monitor `status`; process interruption is recoverable through the journal.

After a complete Spark engineering pilot, record resolved dependency pins:

```bash
docker run --rm --gpus all --network none --entrypoint python \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$HOME/Desktop/eascca-cue-runs,dst=/outputs" \
  qwen-annotation-local:spark /workspace/scripts/freeze_environment.py \
  --pilot-run /outputs/pilot-one --output-dir /outputs/validated-environment
```

Keep NVIDIA's PyTorch/CUDA stack tied to its base image digest; its locally built PyTorch pin may not be installable from PyPI. The captured lockfile is provenance, not an instruction to reinstall every package into a different base. Human cue-quality review remains a separate release requirement.

References checked during implementation:

- [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Transformers Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
- [NVIDIA PyTorch 25.09 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-09.html)
- [NVIDIA DGX Spark software guide](https://docs.nvidia.com/dgx/dgx-spark/software.html)

