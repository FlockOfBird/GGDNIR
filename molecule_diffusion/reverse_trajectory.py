"""Project actual reverse-diffusion sample paths into a clean-reference PCA basis.

The diagnostic records the same latent samples throughout the ordinary stochastic
DDPM sampler.  PCA is fitted only on clean, mask-mean-pooled reference latents;
no noisy or generated state is allowed to influence the visualization axes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .latent import LatentStandardizer
from .main import (
    artifact_paths,
    instantiate_diffusion,
    instantiate_learned_projector,
)
from .projection import LearnedTokenProjector, projection_metadata_compatible
from .sampling import reverse_diffusion_step
from .utils import masked_mean, resolve_device, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Model run containing checkpoints/diffusion.pt.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help=(
            "Directory containing generative_embedding_cache.pt. Defaults to "
            "--artifact-dir. Learned-projection runs must point to their raw cache."
        ),
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Clean cache split used to fit the PCA reference (default: val).",
    )
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=2048,
        help="Number of clean molecules used for PCA and manifold distances.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=24,
        help="Number of actual reverse paths to record (default: 24).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Reverse-sampling batch size (default: 12).",
    )
    parser.add_argument(
        "--reference-batch-size",
        type=int,
        default=128,
        help="Clean-reference projection batch size (default: 128).",
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=16,
        help=(
            "Number of states to retain, including initial noise and final x0 "
            "(default: 16). Ignored when --save-steps is supplied."
        ),
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Reverse denoiser steps to save after they execute. Initial noise is "
            "always saved; step 0 must be included so the final x0 is retained."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device; unavailable CUDA falls back to CPU (default: cuda).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory. Defaults to <artifact-dir>/diagnostics/diffusion/"
            "projected_reverse_trajectories."
        ),
    )
    parser.add_argument(
        "--allow-full-cache-load",
        action="store_true",
        help="Allow a non-memory-mapped cache load if mmap is unavailable.",
    )
    return parser


def select_save_steps(
    num_steps: int,
    num_checkpoints: int,
    requested_steps: list[int] | None = None,
) -> list[int]:
    """Return unique denoiser steps in reverse-execution order."""
    if num_steps <= 0:
        raise ValueError("The diffusion schedule must contain at least one step.")
    if requested_steps is not None:
        invalid = sorted({step for step in requested_steps if not 0 <= step < num_steps})
        if invalid:
            raise ValueError(
                f"--save-steps must be in [0, {num_steps - 1}]; invalid: {invalid}"
            )
        if 0 not in requested_steps:
            raise ValueError("--save-steps must include step 0 to retain final x0.")
        return sorted(set(requested_steps), reverse=True)

    if num_checkpoints < 2:
        raise ValueError("--num-checkpoints must be at least two.")
    saved_reverse_steps = np.linspace(
        num_steps - 1,
        0,
        num=max(1, num_checkpoints - 1),
        dtype=np.int64,
    )
    return sorted({int(value) for value in saved_reverse_steps} | {0}, reverse=True)


def fit_clean_pca(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a two-component low-rank PCA to clean pooled reference vectors."""
    reference = torch.as_tensor(reference, dtype=torch.float32, device="cpu")
    if reference.ndim != 2 or min(reference.shape) < 2:
        raise ValueError("PCA requires at least two clean vectors and two features.")
    mean = reference.mean(dim=0)
    centered = reference - mean
    rank = min(8, *centered.shape)
    _, singular_values, right_vectors = torch.pca_lowrank(
        centered,
        q=rank,
        center=False,
        niter=4,
    )
    components = right_vectors[:, :2].transpose(0, 1)
    total = centered.double().square().sum().clamp_min(1e-12)
    explained = singular_values[:2].double().square() / total
    return mean, components, explained.float()


def project_pca(
    values: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32, device="cpu")
    return (values - mean) @ components.transpose(0, 1)


def per_sequence_rms(
    latents: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    valid = attention_mask.unsqueeze(-1).to(dtype=latents.dtype)
    coordinate_count = (
        attention_mask.sum(dim=1).clamp_min(1) * latents.size(-1)
    ).to(dtype=latents.dtype)
    return ((latents * valid).square().sum(dim=(1, 2)) / coordinate_count).sqrt()


def nearest_reference_distance(
    pooled: torch.Tensor,
    reference: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Euclidean nearest-reference distance normalized by sqrt(latent_dim)."""
    pooled = pooled.float().cpu()
    reference = reference.float().cpu()
    scale = float(reference.size(1)) ** 0.5
    minima = []
    for start in range(0, pooled.size(0), chunk_size):
        distances = torch.cdist(pooled[start : start + chunk_size], reference)
        minima.append(distances.min(dim=1).values / scale)
    return torch.cat(minima)


def _load_cache(path: Path, allow_full_load: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing generative embedding cache: {path}. Run prepare-embeddings first."
        )
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError) as error:
        if not allow_full_load:
            raise RuntimeError(
                f"Could not memory-map {path}. Upgrade PyTorch or pass "
                "--allow-full-cache-load only when enough host RAM is available."
            ) from error
        return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_reference_transform(
    payload: dict,
    paths: dict[str, Path],
    schedule,
    denoiser,
    device: torch.device,
):
    """Validate coordinates and return a token transform into diffusion space."""
    diffusion_projection = schedule.latent_projection
    cache_projection = payload.get("latent_projection")
    cache_dim = int(payload["latent_dim"])

    if (
        isinstance(diffusion_projection, dict)
        and diffusion_projection.get("kind") == LearnedTokenProjector.KIND
    ):
        if cache_projection is not None:
            raise ValueError("Learned projection requires an unprojected raw cache.")
        if not paths["learned_projector"].is_file():
            raise FileNotFoundError(
                f"Missing learned projector: {paths['learned_projector']}"
            )
        projector = instantiate_learned_projector(
            paths["learned_projector"], device, diffusion_projection
        )
        projector.eval()
        if projector.input_dim != cache_dim:
            raise ValueError("Raw cache dimension does not match the learned projector input.")
        if projector.output_dim != denoiser.config.latent_dim:
            raise ValueError("Learned projector output does not match diffusion dimension.")

        def transform(latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            projected = projector(latents, mask)
            return schedule.standardize_latents(projected) * mask.unsqueeze(-1).to(
                projected.dtype
            )

        return transform, "learned_projector"

    if not projection_metadata_compatible(diffusion_projection, cache_projection):
        raise ValueError("Embedding cache and diffusion checkpoint projections do not match.")
    if cache_dim != denoiser.config.latent_dim:
        raise ValueError("Embedding cache and diffusion checkpoint dimensions do not match.")
    cache_standardizer = LatentStandardizer.from_metadata(
        payload.get("latent_standardization"), latent_dim=cache_dim
    ).to(device)
    if not cache_standardizer.is_compatible_with(schedule.latent_standardizer):
        raise ValueError("Embedding cache and diffusion standardizations do not match.")

    def transform(latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        standardized = schedule.standardize_latents(latents)
        return standardized * mask.unsqueeze(-1).to(standardized.dtype)

    return transform, "cache_coordinates"


@torch.inference_mode()
def load_clean_reference(
    split: dict,
    indices: torch.Tensor,
    transform,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    pooled_batches = []
    for start in tqdm(
        range(0, len(indices), batch_size),
        desc="clean-reference",
        unit="batch",
    ):
        batch_indices = indices[start : start + batch_size]
        latents = split["sequence_embeddings"][batch_indices].to(
            device=device, dtype=torch.float32
        )
        mask = split["attention_mask"][batch_indices].to(
            device=device, dtype=torch.bool
        )
        diffusion_latents = transform(latents, mask)
        pooled_batches.append(masked_mean(diffusion_latents, mask).cpu())
    return torch.cat(pooled_batches)


@torch.inference_mode()
def sample_paths(
    denoiser,
    schedule,
    num_trajectories: int,
    batch_size: int,
    save_steps: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pooled states, RMS values, masks, and reverse iterations."""
    pooled_batches: list[torch.Tensor] = []
    rms_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    save_set = set(save_steps)
    expected_iterations = torch.tensor(
        [0, *[schedule.num_steps - step for step in save_steps]],
        dtype=torch.int64,
    )

    for start in tqdm(
        range(0, num_trajectories, batch_size),
        desc="reverse-trajectories",
        unit="batch",
    ):
        current_size = min(batch_size, num_trajectories - start)
        mask = schedule.sample_attention_mask(current_size, device)
        valid = mask.unsqueeze(-1).to(dtype=torch.float32)
        latents = torch.randn(
            current_size,
            denoiser.config.max_sequence_length,
            denoiser.config.latent_dim,
            device=device,
            dtype=torch.float32,
        ) * valid

        pooled_states = [masked_mean(latents, mask).cpu()]
        rms_states = [per_sequence_rms(latents, mask).cpu()]
        for step in reversed(range(schedule.num_steps)):
            latents = reverse_diffusion_step(
                latents=latents,
                attention_mask=mask,
                step=step,
                denoiser=denoiser,
                schedule=schedule,
                guidance_regressor=None,
                guidance_scale=0.0,
                guidance_gamma=1.0,
            )
            if step in save_set:
                pooled_states.append(masked_mean(latents, mask).cpu())
                rms_states.append(per_sequence_rms(latents, mask).cpu())

        pooled_batch = torch.stack(pooled_states, dim=0)
        rms_batch = torch.stack(rms_states, dim=0)
        if pooled_batch.size(0) != expected_iterations.numel():
            raise RuntimeError("A requested reverse checkpoint was not recorded.")
        pooled_batches.append(pooled_batch)
        rms_batches.append(rms_batch)
        mask_batches.append(mask.cpu())

    return (
        torch.cat(pooled_batches, dim=1),
        torch.cat(rms_batches, dim=1),
        torch.cat(mask_batches, dim=0),
        expected_iterations,
    )


def build_records(
    pooled_states: torch.Tensor,
    state_rms: torch.Tensor,
    masks: torch.Tensor,
    reverse_iterations: torch.Tensor,
    save_steps: list[int],
    reference: torch.Tensor,
    pca_mean: torch.Tensor,
    pca_components: torch.Tensor,
    num_steps: int,
) -> tuple[pd.DataFrame, torch.Tensor]:
    checkpoints, trajectories, latent_dim = pooled_states.shape
    flat = pooled_states.reshape(-1, latent_dim)
    projected = project_pca(flat, pca_mean, pca_components).reshape(
        checkpoints, trajectories, 2
    )
    nearest = nearest_reference_distance(flat, reference).reshape(
        checkpoints, trajectories
    )
    sequence_lengths = masks.sum(dim=1).cpu()
    denoiser_steps: list[int | None] = [None, *save_steps]
    rows = []
    for checkpoint_index in range(checkpoints):
        iteration = int(reverse_iterations[checkpoint_index])
        for sample_index in range(trajectories):
            rows.append(
                {
                    "sample_index": sample_index,
                    "checkpoint_index": checkpoint_index,
                    "reverse_iteration": iteration,
                    "reverse_progress": iteration / num_steps,
                    "denoiser_step_just_executed": denoiser_steps[checkpoint_index],
                    "state": (
                        "initial_noise"
                        if checkpoint_index == 0
                        else ("final_x0" if denoiser_steps[checkpoint_index] == 0 else "reverse_state")
                    ),
                    "sequence_length": int(sequence_lengths[sample_index]),
                    "pc1": float(projected[checkpoint_index, sample_index, 0]),
                    "pc2": float(projected[checkpoint_index, sample_index, 1]),
                    "nearest_clean_distance": float(
                        nearest[checkpoint_index, sample_index]
                    ),
                    "pooled_l2_norm": float(
                        pooled_states[checkpoint_index, sample_index].norm()
                    ),
                    "state_rms": float(state_rms[checkpoint_index, sample_index]),
                }
            )
    return pd.DataFrame(rows), projected


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for checkpoint_index, group in records.groupby("checkpoint_index", sort=True):
        row = {
            "checkpoint_index": int(checkpoint_index),
            "reverse_iteration": int(group["reverse_iteration"].iloc[0]),
            "reverse_progress": float(group["reverse_progress"].iloc[0]),
            "denoiser_step_just_executed": group[
                "denoiser_step_just_executed"
            ].iloc[0],
            "state": group["state"].iloc[0],
        }
        for metric in ("nearest_clean_distance", "pooled_l2_norm", "state_rms"):
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_p10"] = float(np.quantile(values, 0.10))
            row[f"{metric}_p90"] = float(np.quantile(values, 0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def _reference_limits(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.005, 0.995])
    width = max(float(upper - lower), 1e-6)
    return float(lower - 0.18 * width), float(upper + 0.18 * width)


def save_path_plot(
    reference_projected: torch.Tensor,
    trajectories_projected: torch.Tensor,
    reverse_iterations: torch.Tensor,
    num_steps: int,
    explained: torch.Tensor,
    output_dir: Path,
) -> Path:
    path = output_dir / "projected_reverse_trajectories.png"
    reference = reference_projected.numpy()
    trajectories = trajectories_projected.numpy()
    progress = reverse_iterations.numpy() / num_steps
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("viridis")

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis in axes:
        axis.scatter(
            reference[:, 0],
            reference[:, 1],
            s=9,
            color="#777777",
            alpha=0.16,
            linewidths=0,
            label="Clean reference",
            rasterized=True,
        )
        for sample_index in range(trajectories.shape[1]):
            points = trajectories[:, sample_index]
            segments = np.stack([points[:-1], points[1:]], axis=1)
            collection = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=1.2,
                alpha=0.58,
            )
            collection.set_array((progress[:-1] + progress[1:]) / 2.0)
            axis.add_collection(collection)
        axis.scatter(
            trajectories[0, :, 0],
            trajectories[0, :, 1],
            marker="*",
            s=46,
            color=cmap(0.0),
            edgecolors="white",
            linewidths=0.35,
            label="Initial noise",
            zorder=3,
        )
        axis.scatter(
            trajectories[-1, :, 0],
            trajectories[-1, :, 1],
            marker="o",
            s=30,
            color=cmap(1.0),
            edgecolors="white",
            linewidths=0.45,
            label="Final $x_0$",
            zorder=4,
        )
        axis.set_xlabel(f"Clean-reference PC1 ({100 * float(explained[0]):.1f}% variance)")
        axis.set_ylabel(f"Clean-reference PC2 ({100 * float(explained[1]):.1f}% variance)")
        axis.grid(alpha=0.18)

    axes[0].set_title("Full reverse paths")
    axes[0].legend(loc="best")
    axes[1].set_title("Clean-manifold zoom")
    axes[1].set_xlim(*_reference_limits(reference[:, 0]))
    axes[1].set_ylim(*_reference_limits(reference[:, 1]))

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axes, pad=0.02, fraction=0.035)
    colorbar.set_label("Reverse progress (0 = initial noise, 1 = final $x_0$)")
    figure.suptitle(
        "Actual unguided reverse-diffusion paths in a clean-reference PCA basis"
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def save_metric_plot(summary: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "reverse_trajectory_metrics.png"
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    x = summary["reverse_progress"].to_numpy()
    specifications = [
        ("nearest_clean_distance", "Nearest clean distance / $\\sqrt{D}$"),
        ("state_rms", "Token-coordinate RMS"),
        ("pooled_l2_norm", "Mean-pooled L2 norm"),
    ]
    for axis, (metric, label) in zip(axes, specifications):
        median = summary[f"{metric}_median"].to_numpy()
        p10 = summary[f"{metric}_p10"].to_numpy()
        p90 = summary[f"{metric}_p90"].to_numpy()
        axis.plot(x, median, color="#1769aa", linewidth=2.0)
        axis.fill_between(x, p10, p90, color="#1769aa", alpha=0.2)
        axis.set_xlabel("Reverse progress")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    figure.suptitle("Full-space diagnostics along the recorded reverse paths")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def run(args: argparse.Namespace) -> dict:
    if args.reference_samples < 2:
        raise ValueError("--reference-samples must be at least two.")
    if (
        args.num_trajectories <= 0
        or args.batch_size <= 0
        or args.reference_batch_size <= 0
    ):
        raise ValueError(
            "--num-trajectories, --batch-size, and --reference-batch-size must be positive."
        )

    set_seed(args.seed)
    device = resolve_device(args.device)
    paths = artifact_paths(args.artifact_dir, args.cache_dir)
    if not paths["diffusion"].is_file():
        raise FileNotFoundError(f"Missing diffusion checkpoint: {paths['diffusion']}")
    payload = _load_cache(paths["generative_embedding_cache"], args.allow_full_cache_load)
    if args.split not in payload.get("splits", {}):
        raise KeyError(f"Cache does not contain split {args.split!r}.")
    split = payload["splits"][args.split]
    available = len(split["smiles"])
    if args.reference_samples > available:
        raise ValueError(
            f"Requested {args.reference_samples} reference samples, but "
            f"{args.split} contains {available}."
        )

    print(f"Loading diffusion checkpoint: {paths['diffusion']}")
    denoiser, schedule = instantiate_diffusion(paths["diffusion"], device)
    if denoiser.config.max_sequence_length != int(payload["sequence_length"]):
        raise ValueError("Embedding cache and diffusion sequence lengths do not match.")
    transform, reference_transform = _resolve_reference_transform(
        payload, paths, schedule, denoiser, device
    )
    save_steps = select_save_steps(
        schedule.num_steps, args.num_checkpoints, args.save_steps
    )

    index_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    reference_indices = torch.randperm(available, generator=index_generator)[
        : args.reference_samples
    ]
    reference = load_clean_reference(
        split, reference_indices, transform, args.reference_batch_size, device
    )
    pca_mean, pca_components, explained = fit_clean_pca(reference)
    reference_projected = project_pca(reference, pca_mean, pca_components)

    pooled_states, state_rms, masks, reverse_iterations = sample_paths(
        denoiser=denoiser,
        schedule=schedule,
        num_trajectories=args.num_trajectories,
        batch_size=args.batch_size,
        save_steps=save_steps,
        device=device,
    )
    records, projected = build_records(
        pooled_states=pooled_states,
        state_rms=state_rms,
        masks=masks,
        reverse_iterations=reverse_iterations,
        save_steps=save_steps,
        reference=reference,
        pca_mean=pca_mean,
        pca_components=pca_components,
        num_steps=schedule.num_steps,
    )
    summary = summarize_records(records)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.artifact_dir)
        / "diagnostics"
        / "diffusion"
        / "projected_reverse_trajectories"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "trajectory_points.csv"
    summary_path = output_dir / "trajectory_summary.csv"
    arrays_path = output_dir / "trajectory_arrays.npz"
    metrics_path = output_dir / "metrics.json"
    records.to_csv(records_path, index=False)
    summary.to_csv(summary_path, index=False)
    np.savez_compressed(
        arrays_path,
        clean_reference=reference.numpy(),
        clean_reference_projected=reference_projected.numpy(),
        pooled_states=pooled_states.numpy(),
        projected_states=projected.numpy(),
        attention_masks=masks.numpy(),
        pca_mean=pca_mean.numpy(),
        pca_components=pca_components.numpy(),
        pca_explained_variance_ratio=explained.numpy(),
        reverse_iterations=reverse_iterations.numpy(),
        denoiser_steps=np.asarray([-1, *save_steps], dtype=np.int64),
    )
    path_plot = save_path_plot(
        reference_projected,
        projected,
        reverse_iterations,
        schedule.num_steps,
        explained,
        output_dir,
    )
    metric_plot = save_metric_plot(summary, output_dir)

    metrics = {
        "diagnostic": "projected_reverse_trajectories",
        "artifact_dir": str(args.artifact_dir),
        "cache_path": str(paths["generative_embedding_cache"]),
        "diffusion_checkpoint": str(paths["diffusion"]),
        "learned_projector": (
            str(paths["learned_projector"])
            if reference_transform == "learned_projector"
            else None
        ),
        "split": args.split,
        "reference_samples": args.reference_samples,
        "num_trajectories": args.num_trajectories,
        "batch_size": args.batch_size,
        "reference_batch_size": args.reference_batch_size,
        "diffusion_steps": schedule.num_steps,
        "saved_denoiser_steps": save_steps,
        "reverse_iterations": reverse_iterations.tolist(),
        "device": str(device),
        "seed": args.seed,
        "sampling": "actual stochastic unguided DDPM reverse process",
        "pooling": "attention-mask-aware token mean",
        "pca_fit": "clean reference only",
        "pca_explained_variance_ratio": explained.tolist(),
        "coordinate_space": getattr(schedule, "latent_coordinate_space", None),
        "reference_transform": reference_transform,
        "nearest_distance": "Euclidean distance in full diffusion space divided by sqrt(D)",
        "outputs": {
            "path_plot": str(path_plot),
            "metric_plot": str(metric_plot),
            "trajectory_points": str(records_path),
            "trajectory_summary": str(summary_path),
            "trajectory_arrays": str(arrays_path),
            "metrics": str(metrics_path),
        },
    }
    save_json(metrics_path, metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = run(args)
    print("\nProjected reverse-trajectory diagnostic complete")
    print(json.dumps(metrics["outputs"], indent=2))


if __name__ == "__main__":
    main()
