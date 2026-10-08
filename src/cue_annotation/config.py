from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .util import digest


@dataclass(frozen=True)
class Config:
    fps: int = 25
    window_frames: int = 250
    primary_frames: int = 25
    context_frames: int = 5
    batch_size: int = 1
    num_workers: int = 0
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_revision: str = "main"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    min_pixels: int = 4 * 28 * 28
    global_max_pixels: int = 1024 * 28 * 28
    crop_max_pixels: int = 512 * 28 * 28
    input_token_budget: int = 57344
    max_new_tokens: int = 6144
    context_budget: int = 128000
    localisation_score: float = 0.90
    localisation_margin: float = 0.05
    localisation_scales: list[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    global_cache_frames: int = 40
    session_start_frames: dict[str, int] = field(
        default_factory=lambda: {
            "5": 0,
            "7": 9225,
            "8": 0,
            "10": 8725,
            "11": 0,
            "14": 85000,
            "15": 0,
        }
    )
    expected_session_windows: dict[str, int] = field(
        default_factory=lambda: {
            "5": 65,
            "7": 209,
            "8": 41,
            "10": 212,
            "11": 61,
            "14": 19,
            "15": 88,
        }
    )

    def validate(self) -> Config:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(item.default, int) and not isinstance(item.default, bool):
                if type(value) is not int:
                    raise ValueError(f"{item.name} must be an integer")
        if (self.fps, self.window_frames, self.batch_size) != (25, 250, 1):
            raise ValueError("The contract requires 25 FPS, 250 positions and batch size 1")
        if not 1 <= self.primary_frames <= 250 or not 0 <= self.context_frames < 250:
            raise ValueError("Invalid primary/context frame budget")
        if self.num_workers != 0:
            raise ValueError("Use num_workers=0; a single process owns the model and journal")
        if self.dtype != "bfloat16" or self.attention != "sdpa":
            raise ValueError("Initial backend requires BF16 and SDPA")
        if (
            self.min_pixels < 4 * 28 * 28
            or min(self.global_max_pixels, self.crop_max_pixels) < self.min_pixels
        ):
            raise ValueError("Invalid pixel bounds")
        if min(self.input_token_budget, self.max_new_tokens) <= 0:
            raise ValueError("Token budgets must be positive")
        if self.input_token_budget + self.max_new_tokens > self.context_budget:
            raise ValueError("Input plus output exceeds context budget")
        if self.global_cache_frames < self.primary_frames + 2 * self.context_frames:
            raise ValueError("Global cache must hold a full input chunk")
        if not 0 <= self.localisation_score <= 1 or not 0 <= self.localisation_margin <= 1:
            raise ValueError("Matching thresholds must be in [0, 1]")
        if not self.localisation_scales or any(
            type(s) not in (int, float) or s <= 0 for s in self.localisation_scales
        ):
            raise ValueError("Localisation scales must be positive")
        for mapping in (self.session_start_frames, self.expected_session_windows):
            if not isinstance(mapping, dict) or any(
                not str(k).isdigit() or type(v) is not int or v < 0 for k, v in mapping.items()
            ):
                raise ValueError("Session mappings require numeric keys and nonnegative integer values")
        if (
            self.model_id not in {"Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"}
            or not self.model_revision
        ):
            raise ValueError("Select the proposed Qwen2.5-VL 7B or 3B checkpoint and a revision")
        return self

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return digest(self.as_dict())


def load_config(path: Path | None = None) -> Config:
    value = json.loads(path.read_text(encoding="utf-8")) if path else {}
    if not isinstance(value, dict):
        raise ValueError("Configuration must be an object")
    unknown = set(value) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return Config(**value).validate()
