"""Inspect the actual Spark GPU environment before a model pilot."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from cue_annotation.qwen_backend import environment
from cue_annotation.util import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "packages": environment(),
        "passed": False,
    }
    try:
        import torch

        if platform.machine().lower() not in {"aarch64", "arm64"}:
            raise RuntimeError("Spark deployment requires ARM64")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA and BF16 must be available")
        result.update(
            torch_cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            compute_capability=list(torch.cuda.get_device_capability(0)),
        )
        tensor = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.bfloat16)
        torch.nn.functional.scaled_dot_product_attention(tensor, tensor, tensor)
        torch.cuda.synchronize()
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        result.update(driver=driver.stdout.strip(), passed=True)
    except (ImportError, RuntimeError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
