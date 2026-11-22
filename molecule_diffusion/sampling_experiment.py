from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .utils import save_json


_EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sampling_experiment_paths(experiment_dir: str | Path) -> dict[str, Path]:
    root = Path(experiment_dir)
    return {
        "root": root,
        "manifest": root / "experiment.json",
        "samples": root / "samples.csv",
        "metrics": root / "metrics.json",
        "plots": root / "plots",
    }


def _validate_experiment_name(name: str) -> str:
    name = str(name).strip()
    if (
        not name
        or name in {".", ".."}
        or len(name) > 160
        or _EXPERIMENT_NAME_PATTERN.fullmatch(name) is None
    ):
        raise ValueError(
            "Experiment names must be 1-160 characters and contain only letters, "
            "numbers, periods, underscores, or hyphens."
        )
    return name


def _scale_slug(guidance_scale: float) -> str:
    return (
        format(float(guidance_scale), ".6g")
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def _automatic_experiment_name(
    property_name: str,
    guidance_scale: float,
    seed: int,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    direction = (
        "increase"
        if guidance_scale > 0.0
        else "decrease"
        if guidance_scale < 0.0
        else "unguided"
    )
    return (
        f"{timestamp}_{property_name}_{direction}_"
        f"scale_{_scale_slug(guidance_scale)}_seed_{seed}"
    )


def create_sampling_experiment(
    artifact_dir: str | Path,
    *,
    property_name: str,
    guidance_scale: float,
    seed: int,
    experiment_name: str = "",
) -> dict[str, Path]:
    parent = Path(artifact_dir) / "experiments" / "sampling"
    parent.mkdir(parents=True, exist_ok=True)
    name = (
        _validate_experiment_name(experiment_name)
        if experiment_name
        else _automatic_experiment_name(property_name, guidance_scale, seed)
    )
    root = parent / name
    try:
        root.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"Sampling experiment already exists: {root}. "
            "Choose another --experiment-name."
        ) from error
    paths = sampling_experiment_paths(root)
    paths["plots"].mkdir()
    return paths


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_experiment_manifest(
    paths: dict[str, Path],
    updates: dict,
) -> dict:
    manifest = {}
    if paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest.update(updates)
    save_json(paths["manifest"], manifest)
    return manifest


def record_latest_sampling_experiment(
    artifact_dir: str | Path,
    experiment_paths: dict[str, Path],
) -> Path:
    parent = Path(artifact_dir) / "experiments" / "sampling"
    pointer_path = parent / "latest.json"
    save_json(
        pointer_path,
        {
            "experiment_name": experiment_paths["root"].name,
            "updated_at": utc_timestamp(),
        },
    )
    return pointer_path


def latest_sampling_experiment(
    artifact_dir: str | Path,
) -> dict[str, Path]:
    parent = Path(artifact_dir) / "experiments" / "sampling"
    pointer_path = parent / "latest.json"
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"No sampling experiment index found at {pointer_path}. "
            "Run sample first or pass --samples-path."
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    name = _validate_experiment_name(pointer.get("experiment_name", ""))
    paths = sampling_experiment_paths(parent / name)
    if not paths["root"].is_dir() or not paths["samples"].is_file():
        raise FileNotFoundError(
            f"Latest sampling experiment is incomplete or missing: {paths['root']}."
        )
    return paths
