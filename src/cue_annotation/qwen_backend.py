from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .util import atomic_json, digest, file_sha256


class SplitRequired(RuntimeError):
    pass


class FatalGPUError(RuntimeError):
    pass


def download_model(config, cache_dir: Path | None = None) -> dict:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the [vlm] extra to download model weights") from exc
    info = HfApi().model_info(config.model_id, revision=config.model_revision)
    path = snapshot_download(
        repo_id=config.model_id,
        revision=info.sha,
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )
    identity = model_identity(Path(path), config)
    atomic_json(Path(path).parent.parent / "eascca_download_manifest.json", identity)
    return identity


def model_identity(model_dir: Path, config) -> dict:
    model_dir = model_dir.expanduser().resolve()
    configuration = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if configuration.get("model_type") != "qwen2_5_vl":
        raise ValueError("The backend requires a Qwen2.5-VL checkpoint")
    architecture = {"Qwen/Qwen2.5-VL-7B-Instruct": (3584, 28), "Qwen/Qwen2.5-VL-3B-Instruct": (2048, 36)}
    if (configuration.get("hidden_size"), configuration.get("num_hidden_layers")) != architecture.get(
        config.model_id
    ):
        raise ValueError("Local checkpoint architecture does not match the configured 7B/3B model")
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError("No safetensors model weights in the local checkpoint")
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        if not names <= {p.name for p in weights}:
            raise FileNotFoundError("Checkpoint weight shards are incomplete")
    files = sorted(
        [
            p
            for p in model_dir.iterdir()
            if p.is_file() and p.suffix in {".json", ".safetensors", ".txt", ".model", ".tiktoken"}
        ],
        key=lambda p: p.name,
    )
    identity = {
        "model_id": config.model_id,
        "resolved_revision": model_dir.name,
        "artifact_fingerprint": digest([[p.name, file_sha256(p)] for p in files]),
        "max_position_embeddings": configuration["max_position_embeddings"],
    }
    if identity["max_position_embeddings"] < config.context_budget:
        raise ValueError("Configured context budget exceeds checkpoint context")
    # Paths are deliberately not part of the identity, permitting the same checkpoint to move.
    identity["local_path"] = str(model_dir)
    return identity


def environment() -> dict:
    result = {}
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "qwen-vl-utils",
        "huggingface-hub",
        "Pillow",
        "pydantic",
        "numpy",
        "opencv-python-headless",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not_installed"
    return result


class QwenBackend:
    def __init__(self, config, model_dir: Path):
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install the [vlm] extra in the validated Spark GPU environment") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen inference requires CUDA; core commands and tests run without a GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected CUDA device does not support BF16")
        self.torch = torch
        self.config = config
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=config.min_pixels,
            max_pixels=config.global_max_pixels,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation=config.attention,
            device_map={"": "cuda:0"},
        )
        self.model.eval()
        self.last_metrics = {}

    def iter_tasks(self, iterator):
        torch = self.torch

        class Tasks(torch.utils.data.IterableDataset):
            def __iter__(self):
                return iterator

        loader = torch.utils.data.DataLoader(
            Tasks(), batch_size=1, num_workers=0, collate_fn=lambda rows: rows[0]
        )
        yield from loader

    def release_memory(self):
        self.torch.cuda.empty_cache()

    def generate(self, messages) -> str:
        torch = self.torch
        inputs = None
        outputs = None
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=images, videos=videos, padding=False, return_tensors="pt"
            )
            input_tokens = inputs["input_ids"].shape[-1]
            if (
                input_tokens > self.config.input_token_budget
                or input_tokens + self.config.max_new_tokens > self.config.context_budget
            ):
                raise SplitRequired(f"Encoded input length {input_tokens} exceeds configured token budget")
            inputs = inputs.to("cuda:0")
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=self.config.max_new_tokens, use_cache=True
                )
            generated = outputs[:, input_tokens:]
            self.last_metrics = {
                "input_tokens": input_tokens,
                "output_tokens": generated.shape[-1],
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            }
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            if generated.shape[-1] >= self.config.max_new_tokens and int(generated[0, -1]) not in eos:
                raise SplitRequired("Model output reached generation limit without EOS")
            return self.processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        except torch.cuda.OutOfMemoryError as exc:
            raise SplitRequired("CUDA out of memory") from exc
        except RuntimeError as exc:
            if any(
                s in str(exc).lower()
                for s in ("device-side assert", "illegal memory access", "unspecified launch failure")
            ):
                raise FatalGPUError(str(exc)) from exc
            raise
        finally:
            del inputs, outputs
            torch.cuda.empty_cache()
