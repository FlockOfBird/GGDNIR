"""Plot unguided reverse-diffusion paths for completed mean-pooled study runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from sklearn.decomposition import PCA
import torch

from .main import instantiate_diffusion
from .sampling import reverse_diffusion_step


CONDITIONS = (
    ("chemberta", "frozen", "ChemBERTa · frozen"),
    ("chemberta", "scratch", "ChemBERTa · scratch"),
    ("llama", "frozen", "Llama 3.2 1B · frozen"),
)
COLORS = ("#0072B2", "#D55E00", "#009E73")


def _run_paths(root: Path, encoder: str, variant: str, training_seed: int) -> tuple[Path, Path]:
    run = root / "runs" / encoder / variant / "mean_pooled_pca" / f"seed_{training_seed}"
    cache_dir = (
        root / "encoders" / encoder / variant
        / ("shared" if variant == "frozen" else f"seed_{training_seed}")
        / "mean_pooled_pca256"
    )
    return run / "checkpoints" / "diffusion.pt", cache_dir / "generative_embedding_cache.pt"


def _reference_pca(cache_path: Path, schedule, count: int, seed: int) -> tuple[PCA, np.ndarray, np.ndarray]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("latent_representation") != schedule.latent_representation:
        raise ValueError(f"Latent representation mismatch: {cache_path}")
    vectors = payload["splits"]["train"]["embeddings"]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(vectors), size=min(count, len(vectors)), replace=False)
    clean = vectors[indices].to(dtype=torch.float32)
    clean = schedule.latent_standardizer.cpu().standardize(clean).numpy()
    if not np.isfinite(clean).all():
        raise ValueError(f"Non-finite standardized reference embeddings: {cache_path}")
    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    coordinates = pca.fit_transform(clean)
    del payload
    clean_rms = np.sqrt(np.mean(clean.astype(np.float64) ** 2, axis=-1))
    return pca, coordinates, clean_rms


def _select_distant_initial(
    references: list[dict],
    seed: int,
    latent_dim: int,
    radius_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Condition one Gaussian draw to fixed 2D PCA radii in all model views."""
    raw = np.random.default_rng(seed).standard_normal(latent_dim).astype(np.float64)
    components = np.concatenate(
        [item["pca"].components_.astype(np.float64) for item in references], axis=0
    )
    targets = []
    for item in references:
        pca = item["pca"]
        coordinates = (raw - pca.mean_) @ pca.components_.T
        direction = coordinates / np.linalg.norm(coordinates)
        target_coordinates = direction * radius_ratio * item["reference_radius_95"]
        targets.extend(target_coordinates + pca.mean_ @ pca.components_.T)
    targets = np.asarray(targets, dtype=np.float64)
    gram = components @ components.T
    if np.linalg.cond(gram) > 1e8:
        raise ValueError("Model PCA directions are too dependent for a shared constrained start.")
    adjusted = raw + components.T @ np.linalg.solve(
        gram, targets - components @ raw
    )
    initial = adjusted.astype(np.float32)
    ratios = {
        item["label"]: float(
            np.linalg.norm(item["pca"].transform(initial[None, :])[0])
            / item["reference_radius_95"]
        )
        for item in references
    }
    if any(abs(value - radius_ratio) > 1e-4 for value in ratios.values()):
        raise AssertionError("Shared start did not reach the requested PCA radii.")
    return initial, ratios


@torch.inference_mode()
def _sample_paths(
    denoiser,
    schedule,
    checkpoints: int,
    initial_noise: torch.Tensor,
    reverse_noise: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    if schedule.latent_representation != "molecule_mean_pooled_v1":
        raise ValueError("This plot supports only mean-pooled molecule latents.")
    if denoiser.config.max_sequence_length != 1:
        raise ValueError("Mean-pooled trajectories must have sequence length one.")
    device = next(denoiser.parameters()).device
    trajectories = initial_noise.size(0)
    expected_shape = (trajectories, 1, denoiser.config.latent_dim)
    if tuple(initial_noise.shape) != expected_shape or tuple(reverse_noise.shape) != (
        schedule.num_steps - 1, *expected_shape
    ):
        raise ValueError("Shared noise bank does not match this diffusion checkpoint.")
    if initial_noise.device != device or reverse_noise.device != device:
        raise ValueError("Shared noise bank and denoiser must use the same device.")
    mask = torch.ones((trajectories, 1), dtype=torch.bool, device=device)
    latents = initial_noise.clone()
    # Counts include x_T before any denoising and x_0 after all reverse steps.
    counts = np.rint(np.linspace(0, schedule.num_steps, checkpoints)).astype(int)
    if len(np.unique(counts)) != checkpoints:
        raise ValueError("More checkpoints requested than distinct reverse steps.")
    states = [latents[:, 0, :].cpu().numpy().copy()]
    record_at = set(counts[1:].tolist())
    for completed in range(1, schedule.num_steps + 1):
        step = schedule.num_steps - completed
        latents = reverse_diffusion_step(
            latents, mask, step, denoiser, schedule,
            noise=reverse_noise[step - 1] if step > 0 else None,
        )
        if completed in record_at:
            states.append(latents[:, 0, :].cpu().numpy().copy())
    return counts, np.stack(states, axis=0)


def _plot(results: list[dict], output: Path, reference_count: int,
          minimum_radius_ratio: float) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(16.5, 9.5))
    for column, result in enumerate(results):
        top, bottom = axes[:, column]
        reference = result["reference"]
        paths = result["coordinates"]
        progress = result["progress"]
        radius = result["reference_radius_95"]
        central_reference = reference[np.linalg.norm(reference, axis=1) <= radius]
        top.scatter(central_reference[:, 0], central_reference[:, 1], s=3, color="#A8A8A8",
                    alpha=0.24, rasterized=True, zorder=1)
        top.add_patch(Circle((0, 0), radius, fill=False, linestyle="--",
                             linewidth=1.2, edgecolor="#777777", zorder=2))
        for trajectory, color in enumerate(COLORS[:paths.shape[1]]):
            points = paths[:, trajectory, :]
            top.plot(points[:, 0], points[:, 1], color=color, alpha=0.72,
                     linewidth=1.55, zorder=2)
            top.scatter(points[:, 0], points[:, 1], color=color, s=15,
                        edgecolor="white", linewidth=0.3, zorder=3)
            top.scatter(points[-1, 0], points[-1, 1], color=color, marker="*",
                        s=115, edgecolor="black", linewidth=0.4, zorder=5)
        top.scatter(paths[0, 0, 0], paths[0, 0, 1], color="black", marker="X",
                    s=90, edgecolor="white", linewidth=0.6, zorder=6)
        top.set_title(result["label"])
        top.set_xlabel("Reference PC1")
        if column == 0:
            top.set_ylabel("Reference PC2")
            bottom.set_ylabel("Pairwise latent RMS distance")
        limit = 1.12 * max(radius, np.max(np.abs(paths)), np.max(np.abs(central_reference)))
        top.set_xlim(-limit, limit)
        top.set_ylim(-limit, limit)
        for pair_index, ((left, right), color) in enumerate(zip(
            ((0, 1), (0, 2), (1, 2)), COLORS
        )):
            bottom.plot(progress, result["pairwise"][:, pair_index], color=color,
                        marker="o", markersize=3.4, linewidth=1.7,
                        label=f"Runs {left + 1}–{right + 1}")
        bottom.legend(loc="upper left", fontsize=8, framealpha=0.8)
        bottom.set_xlabel("Reverse diffusion completed (%)")
        bottom.set_xlim(0, 100)
        bottom.grid(alpha=0.17)
        bottom.set_ylim(bottom=0)
    legend = [Line2D([0], [0], color=color, marker="o", label=f"DDPM run {i + 1}")
              for i, color in enumerate(COLORS[:results[0]["coordinates"].shape[1]])]
    legend += [Line2D([0], [0], color="#A8A8A8", marker="o", linestyle="none",
                      label=f"Central 95% reference (n={reference_count:,} fitted)"),
               Line2D([0], [0], color="#777777", linestyle="--", label="95% boundary"),
               Line2D([0], [0], color="black", marker="X", linestyle="none", label="Shared initial noise"),
               Line2D([0], [0], color="black", marker="*", linestyle="none", label="Final latent")]
    fig.legend(handles=legend, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.015), fontsize=8.5)
    fig.suptitle("Mean-pooled PCA 256: three DDPM paths from one tail Gaussian start", fontsize=15)
    fig.text(0.5, 0.057,
             f"Shared Gaussian vector conditioned to {minimum_radius_ratio:.1f}× each model's "
             "95% projected reference radius; PCA axes differ across models. Gray points show central 95%.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0.02, 0.095, 1, 0.95), w_pad=2.0, h_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("artifacts/latent_space_study_v1"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--training-seed", type=int, default=101)
    parser.add_argument("--seed", type=int, default=10101)
    parser.add_argument("--checkpoints", type=int, default=16)
    parser.add_argument("--num-trajectories", type=int, default=3)
    parser.add_argument("--reference-count", type=int, default=5000)
    parser.add_argument("--minimum-radius-ratio", type=float, default=1.2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.checkpoints < 2 or args.reference_count < 2 or args.num_trajectories != 3:
        parser.error("This comparison requires 3 trajectories, at least 2 checkpoints, and at least 2 reference samples.")
    if args.minimum_radius_ratio <= 1:
        parser.error("Radius ratio must exceed 1.")
    device = torch.device(args.device)
    output = args.output or args.study_root / "analysis" / "mean_pooled_reverse_diffusion_trajectories.pdf"
    models = []
    for encoder, variant, label in CONDITIONS:
        checkpoint_path, cache_path = _run_paths(args.study_root, encoder, variant, args.training_seed)
        if not checkpoint_path.is_file() or not cache_path.is_file():
            raise FileNotFoundError(f"Missing completed mean-pooled run: {label}")
        print(f"Loading {label}: {checkpoint_path}", flush=True)
        denoiser, schedule = instantiate_diffusion(checkpoint_path, device)
        pca, reference, _ = _reference_pca(cache_path, schedule, args.reference_count, args.seed)
        schedule.to(device)
        reference_radius_95 = float(np.quantile(np.linalg.norm(reference, axis=1), 0.95))
        models.append({"encoder": encoder, "variant": variant, "label": label,
                       "denoiser": denoiser, "schedule": schedule, "pca": pca,
                       "reference": reference, "reference_radius_95": reference_radius_95})
    latent_dim = models[0]["denoiser"].config.latent_dim
    num_steps = models[0]["schedule"].num_steps
    if any(item["denoiser"].config.latent_dim != latent_dim
           or item["schedule"].num_steps != num_steps for item in models):
        raise ValueError("A shared numeric start and reverse noise require matching latent dimensions and DDPM steps.")
    initial_vector, start_ratios = _select_distant_initial(
        models, args.seed, latent_dim, args.minimum_radius_ratio
    )
    initial_noise = torch.from_numpy(initial_vector).to(device).view(1, 1, latent_dim).repeat(
        args.num_trajectories, 1, 1
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    # Runs draw independent reverse noise; each model reuses the same three noise streams.
    reverse_noise = torch.randn(
        (num_steps - 1, args.num_trajectories, 1, latent_dim),
        device=device, generator=generator,
    )
    if torch.equal(reverse_noise[:, 0], reverse_noise[:, 1]):
        raise AssertionError("Repeated DDPM runs unexpectedly received identical reverse noise.")
    initial_fingerprint = hashlib.sha256(initial_vector.tobytes()).hexdigest()
    reverse_fingerprint = hashlib.sha256(reverse_noise.cpu().numpy().tobytes()).hexdigest()
    print(f"Conditioned one shared Gaussian start to these radius ratios: {start_ratios}", flush=True)
    results = []
    table = []
    pairwise_table = []
    pairs = ((0, 1), (0, 2), (1, 2))
    for item in models:
        encoder, variant, label = item["encoder"], item["variant"], item["label"]
        denoiser, schedule, pca = item["denoiser"], item["schedule"], item["pca"]
        print(f"Sampling three DDPM paths for {label}", flush=True)
        counts, states = _sample_paths(
            denoiser, schedule, args.checkpoints, initial_noise, reverse_noise
        )
        if not np.array_equal(states[0], np.repeat(initial_vector[None, :], args.num_trajectories, axis=0)):
            raise AssertionError(f"Initial noise changed for {label}.")
        coordinates = pca.transform(states.reshape(-1, states.shape[-1])).reshape(
            len(counts), args.num_trajectories, 2
        )
        rms = np.sqrt(np.mean(states.astype(np.float64) ** 2, axis=-1))
        pairwise = np.column_stack([
            np.sqrt(np.mean((states[:, left].astype(np.float64)
                             - states[:, right].astype(np.float64)) ** 2, axis=-1))
            for left, right in pairs
        ])
        progress = 100 * counts / schedule.num_steps
        results.append({"label": label, "reference": item["reference"], "coordinates": coordinates,
                        "rms": rms, "pairwise": pairwise, "progress": progress,
                        "reference_radius_95": item["reference_radius_95"],
                        "start_radius_ratio": start_ratios[label]})
        for checkpoint_index, completed in enumerate(counts):
            for trajectory in range(args.num_trajectories):
                table.append({"encoder": encoder, "variant": variant,
                              "trajectory": trajectory + 1, "checkpoint": checkpoint_index + 1,
                              "steps_completed": int(completed), "diffusion_timestep": int(schedule.num_steps - completed),
                              "progress_pct": float(progress[checkpoint_index]),
                              "reference_pc1": float(coordinates[checkpoint_index, trajectory, 0]),
                              "reference_pc2": float(coordinates[checkpoint_index, trajectory, 1]),
                              "latent_rms": float(rms[checkpoint_index, trajectory])})
            for pair_index, (left, right) in enumerate(pairs):
                pairwise_table.append({"encoder": encoder, "variant": variant,
                                       "checkpoint": checkpoint_index + 1,
                                       "steps_completed": int(completed),
                                       "run_a": left + 1, "run_b": right + 1,
                                       "pairwise_latent_rms_distance": float(pairwise[checkpoint_index, pair_index])})
        item["denoiser"] = None
        item["schedule"] = None
        del denoiser, schedule
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _plot(results, output, min(args.reference_count, len(models[0]["reference"])),
          args.minimum_radius_ratio)
    import csv
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    pairwise_path = output.with_name(output.stem + "_pairwise.csv")
    with pairwise_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise_table[0]))
        writer.writeheader()
        writer.writerows(pairwise_table)
    noise_path = output.with_name(output.stem + "_noise.npz")
    np.savez_compressed(
        noise_path,
        initial_noise=initial_vector,
        reverse_noise=reverse_noise.cpu().numpy(),
    )
    metadata = {"models": [result["label"] for result in results],
                "sampling": "unguided DDPM reverse_diffusion_step",
                "training_seed": args.training_seed, "sampling_seed": args.seed,
                "noise_matching": "one identical initial Gaussian vector for all runs and models; independent reverse noise per run, matched across models",
                "start_selection": "one N(0, I) draw conditioned to the requested multiple of each model's 95% reference PCA radius",
                "minimum_radius_ratio": args.minimum_radius_ratio,
                "start_radius_ratios": start_ratios,
                "initial_noise_sha256": initial_fingerprint,
                "reverse_noise_sha256": reverse_fingerprint,
                "checkpoints": args.checkpoints, "num_trajectories": args.num_trajectories,
                "reference_count": min(args.reference_count, len(models[0]["reference"])),
                "projection": "per-model PCA fitted to standardized clean training latents",
                "csv": str(csv_path), "pairwise_csv": str(pairwise_path),
                "noise_npz": str(noise_path)}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
