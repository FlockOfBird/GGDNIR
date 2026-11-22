from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_device(device_preference: str) -> torch.device:
    requested_device = torch.device(device_preference)
    if requested_device.type != "cuda":
        return requested_device
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if requested_device.index is not None and requested_device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested CUDA device {requested_device.index}, but only "
            f"{torch.cuda.device_count()} device(s) are available."
        )
    return requested_device


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def detach_to_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu()


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return summed / denom


def cycle(iterable):
    while True:
        for item in iterable:
            yield item
