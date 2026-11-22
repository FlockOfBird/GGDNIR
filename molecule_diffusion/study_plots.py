"""Replace latent-space study figures with a reproducible comparison suite."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.decomposition import PCA
import torch

from .config import DataConfig
from .data import build_data_bundle
from .properties import molecule_property
from .study_evaluation import LABELS, PROPERTIES, describe_smiles


INTERFACES = ("pca", "learned", "mean_pooled_pca")
INTERFACE_LABELS = {
    "pca": "Token PCA 256",
    "learned": "Token learned 256",
    "mean_pooled_pca": "Molecule PCA 256",
}
INTERFACE_COLORS = {
    "pca": "#0072B2",
    "learned": "#D55E00",
    "mean_pooled_pca": "#009E73",
}
ENCODER_ORDER = (("chemberta", "frozen"), ("chemberta", "scratch"),
                 ("llama", "frozen"), ("smiles_roberta", "scratch"))
ENCODER_COLORS = {
    ("chemberta", "frozen"): "#0072B2",
    ("chemberta", "scratch"): "#D55E00",
    ("llama", "frozen"): "#7B3294",
    ("smiles_roberta", "scratch"): "#009E73",
}


def _encoder_label(encoder: str, variant: str) -> str:
    name = {"chemberta": "ChemBERTa", "llama": "Llama 3.2 1B",
            "smiles_roberta": "Small SMILES RoBERTa"}.get(encoder, encoder)
    return f"{name} {variant}"


def _condition_label(row) -> str:
    return f"{_encoder_label(row.encoder, row.variant)} · {INTERFACE_LABELS.get(row.interface, row.interface)}"


def _short_label(row) -> str:
    model = {"chemberta": "CB", "llama": "LL", "smiles_roberta": "SR"}.get(row.encoder, row.encoder)
    state = "S" if row.variant == "scratch" else "F"
    interface = {"pca": "P", "learned": "L", "mean_pooled_pca": "MP"}.get(
        row.interface, row.interface
    )
    return f"{model}-{state}-{interface}"


def _save(fig, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_records(root: Path, metrics: pd.DataFrame) -> list[dict]:
    records = []
    for path in sorted((root / "runs").glob("*/*/*/seed_*/condition.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        interface = manifest.get("condition_projection") or (
            "mean_pooled_pca" if manifest.get("latent_representation") == "mean_pooled"
            else manifest["projection"]
        )
        key = (manifest["encoder"], manifest["variant"], interface,
               manifest["training_seed"])
        records.append({"key": key, "manifest": manifest, "run_dir": path.parent})
    metric_keys = {
        (row.encoder, row.variant, row.interface, row.training_seed)
        for row in metrics.itertuples(index=False)
    }
    record_keys = {record["key"] for record in records}
    if not records or metric_keys != record_keys or len(metric_keys) != len(records):
        raise ValueError(
            "Analysis metrics do not match completed run manifests. "
            "Run molecule_diffusion.study_evaluation first."
        )
    return records


def _reference_properties(dataset_path: str, data_seed: int, reference_size: int) -> np.ndarray:
    config = DataConfig(dataset_path=dataset_path, label_column=None,
                        property_name=None, seed=data_seed, split_strategy="scaffold")
    splits = build_data_bundle(config)
    smiles = splits.train[config.smiles_column].astype(str).tolist()
    rng = np.random.default_rng(data_seed + 2026)
    indices = rng.permutation(len(smiles))[:reference_size]
    values, _, _ = describe_smiles([smiles[index] for index in indices])
    if len(values) != reference_size:
        raise ValueError("The reference set contains invalid SMILES.")
    return values


def _generated_properties(records: list[dict], metrics: pd.DataFrame) -> dict[tuple, np.ndarray]:
    matrices = {}
    for record in records:
        sample_path = Path(record["manifest"]["sample_path"])
        if not sample_path.is_absolute():
            sample_path = Path.cwd() / sample_path
        samples = pd.read_csv(sample_path)["SMILES"].fillna("").astype(str).tolist()
        values, _, _ = describe_smiles(samples)
        match = metrics.loc[
            (metrics.encoder == record["key"][0])
            & (metrics.variant == record["key"][1])
            & (metrics.interface == record["key"][2])
            & (metrics.training_seed == record["key"][3])
        ]
        if len(values) != int(match.iloc[0]["valid"]):
            raise ValueError(f"Valid sample count changed: {sample_path}")
        matrices[record["key"]] = values
    return matrices


def _plot_property_distributions(reference: np.ndarray, matrices: dict[tuple, np.ndarray],
                                 directory: Path) -> None:
    encoders = sorted({key[0] for key in matrices})
    fig, axes = plt.subplots(4, len(encoders), figsize=(7 * len(encoders), 13.5),
                             sharex="row", sharey="row", squeeze=False)
    for prop_index, prop in enumerate(PROPERTIES):
        lower, upper = np.quantile(reference[:, prop_index], [0.005, 0.995])
        pad = 0.1 * (upper - lower)
        bins = np.linspace(lower - pad, upper + pad, 40)
        for column, encoder in enumerate(encoders):
            axis = axes[prop_index, column]
            axis.hist(reference[:, prop_index], bins=bins,
                      weights=np.full(len(reference), 1.0 / len(reference)),
                      histtype="step", color="black", linewidth=2.4)
            for key, values in sorted(matrices.items()):
                if key[0] != encoder:
                    continue
                color = INTERFACE_COLORS.get(key[2], "#777777")
                axis.hist(values[:, prop_index], bins=bins,
                          weights=np.full(len(values), 1.0 / len(values)),
                          histtype="step", color=color,
                          linestyle="--" if key[1] == "scratch" else "-",
                          linewidth=1.6)
            axis.set_xlim(bins[0], bins[-1])
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel(f"{LABELS[prop]}\nProbability per bin")
            if prop_index == 0:
                axis.set_title(_encoder_label(encoder, "").strip())
            if prop_index == 3:
                axis.set_xlabel(LABELS[prop])
    handles = [Line2D([0], [0], color="black", lw=2.4, label="Reference train")]
    for interface in INTERFACES:
        for variant in ("frozen", "scratch"):
            if not any(key[2] == interface and key[1] == variant for key in matrices):
                continue
            handles.append(Line2D(
                [0], [0], color=INTERFACE_COLORS[interface],
                ls="--" if variant == "scratch" else "-", lw=1.8,
                label=f"{variant.capitalize()} {INTERFACE_LABELS[interface]}",
            ))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Generated property distributions vs training reference")
    fig.text(0.5, 0.008,
             "Shared bins and axes by property. Display range follows reference 0.5–99.5% quantiles; scores use all valid samples.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    _save(fig, directory, "01_property_distributions")


def _plot_error_heatmap(metrics: pd.DataFrame, directory: Path) -> None:
    frame = metrics.sort_values("mean_normalized_w1").reset_index(drop=True)
    columns = [f"w1_{name}" for name in PROPERTIES] + ["mean_normalized_w1"]
    values = frame[columns].to_numpy(dtype=float)
    labels = [_condition_label(row) for row in frame.itertuples(index=False)]
    fig, axis = plt.subplots(figsize=(11.5, max(5, 0.6 * len(frame) + 1.4)))
    image = axis.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0,
                        vmax=float(np.max(values)))
    axis.set_xticks(range(5), ["MW", "logP", "QED", "SA", "Mean"])
    axis.set_yticks(range(len(frame)), labels)
    for row in range(len(frame)):
        for column in range(5):
            color = "white" if values[row, column] > np.max(values) * 0.62 else "#202020"
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center",
                      va="center", fontsize=9, color=color)
    axis.axvline(3.5, color="white", linewidth=2)
    axis.set_title("Normalized property distribution error (lower is better)")
    fig.colorbar(image, ax=axis, label="Wasserstein distance / reference IQR", shrink=0.85)
    fig.tight_layout()
    _save(fig, directory, "02_property_error_heatmap")


def _plot_joint_properties(reference: np.ndarray, matrices: dict[tuple, np.ndarray],
                           metrics: pd.DataFrame, directory: Path) -> None:
    keys = [tuple(row) for row in metrics.sort_values("mean_normalized_w1")[
        ["encoder", "variant", "interface", "training_seed"]
    ].itertuples(index=False, name=None)]
    datasets = [("Reference train", reference)] + [
        (_condition_label(metrics.loc[
            (metrics.encoder == key[0]) & (metrics.variant == key[1])
            & (metrics.interface == key[2]) & (metrics.training_seed == key[3])
        ].iloc[0]), matrices[key]) for key in keys
    ]
    qed_index = PROPERTIES.index("qed")
    sa_index = PROPERTIES.index("sas")
    xlo, xhi = np.quantile(reference[:, qed_index], [0.005, 0.995])
    ylo, yhi = np.quantile(reference[:, sa_index], [0.005, 0.995])
    xbins = np.linspace(xlo - 0.08 * (xhi - xlo), xhi + 0.08 * (xhi - xlo), 36)
    ybins = np.linspace(ylo - 0.08 * (yhi - ylo), yhi + 0.08 * (yhi - ylo), 36)
    histograms = [np.histogram2d(values[:, qed_index], values[:, sa_index],
                                 bins=(xbins, ybins))[0] for _, values in datasets]
    norm = LogNorm(vmin=1, vmax=max(float(hist.max()) for hist in histograms))
    columns = min(5, len(datasets))
    rows = int(np.ceil(len(datasets) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.5 * rows),
                             sharex=True, sharey=True, squeeze=False)
    for axis, (label, _), hist in zip(axes.flat, datasets, histograms):
        mesh = axis.pcolormesh(xbins, ybins, hist.T, cmap="viridis", norm=norm,
                               shading="auto")
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("QED")
        axis.set_ylabel("Synthetic accessibility")
    for axis in list(axes.flat)[len(datasets):]:
        axis.set_visible(False)
    fig.suptitle("Joint QED–synthetic accessibility distribution")
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.09, top=0.88,
                        wspace=0.14, hspace=0.30)
    color_axis = fig.add_axes([0.91, 0.18, 0.014, 0.62])
    fig.colorbar(mesh, cax=color_axis, label="Molecules per bin (log scale)")
    _save(fig, directory, "03_qed_sa_joint_distribution")


def _plot_training_curves(records: list[dict], directory: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col")
    for column, interface in enumerate(INTERFACES):
        axes[0, column].set_title(INTERFACE_LABELS[interface])
        for record in records:
            encoder, variant, current_interface, _ = record["key"]
            if current_interface != interface:
                continue
            payload = json.loads((record["run_dir"] / "metrics.json").read_text())
            history = payload["diffusion"]["history"]
            epochs = [item["epoch"] for item in history]
            color = ENCODER_COLORS.get((encoder, variant), "#777777")
            label = _encoder_label(encoder, variant)
            axes[0, column].plot(epochs, [item["val_diffusion_loss"] for item in history],
                                 color=color, lw=1.8, label=label)
            axes[1, column].plot(epochs, [item["val_token_accuracy"] for item in history],
                                 color=color, lw=1.8, label=label)
        for row in range(2):
            axes[row, column].grid(alpha=0.2)
        axes[1, column].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Validation diffusion loss")
    axes[1, 0].set_ylabel("Validation SELFIES token accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Training trajectories by latent interface")
    fig.text(0.5, 0.01, "Compare loss curves within an interface; molecule and token inputs have different shapes.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    _save(fig, directory, "04_training_curves")


def _plot_cost_tradeoff(metrics: pd.DataFrame, directory: Path) -> pd.DataFrame:
    frame = metrics.copy()
    frame["diffusion_training_hours"] = frame["train_seconds_this_invocation"] / 3600
    frame["sampling_minutes"] = frame["sampling_seconds_this_invocation"] / 60
    frame["selected_cache_gb"] = frame["selected_cache_bytes"] / 1e9
    frame["condition_code"] = [_short_label(row) for row in frame.itertuples(index=False)]
    columns = ["encoder", "variant", "interface", "training_seed", "condition_code",
               "mean_normalized_w1", "diffusion_training_hours", "sampling_minutes",
               "selected_cache_gb", "encoder_adaptation_seconds", "raw_cache_seconds",
               "pooled_cache_seconds", "pca_seconds"]
    costs = frame[columns].sort_values("mean_normalized_w1")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    offsets = {"pca": (5, 8), "learned": (5, -10), "mean_pooled_pca": (5, 2)}
    for row in frame.itertuples(index=False):
        color = ENCODER_COLORS.get((row.encoder, row.variant), "#777777")
        marker = {"pca": "o", "learned": "s", "mean_pooled_pca": "D"}.get(
            row.interface, "x"
        )
        for axis, x in ((axes[0], row.diffusion_training_hours),
                        (axes[1], row.selected_cache_gb)):
            axis.scatter(x, row.mean_normalized_w1, color=color, marker=marker,
                         s=90, edgecolors="black", linewidths=0.4, zorder=3)
            axis.annotate(row.condition_code, (x, row.mean_normalized_w1),
                          xytext=offsets.get(row.interface, (5, 4)),
                          textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("Diffusion training elapsed time (hours)")
    axes[0].set_xlim(frame.diffusion_training_hours.min() - 0.6,
                     frame.diffusion_training_hours.max() + 1.2)
    axes[1].set_xlabel("Selected embedding cache (GB)")
    axes[1].set_xscale("log")
    axes[1].set_xlim(frame.selected_cache_gb.min() / 1.3,
                     frame.selected_cache_gb.max() * 1.6)
    for axis in axes:
        axis.set_ylabel("Mean normalized W1 (lower is better)")
        axis.grid(alpha=0.2)
    fig.suptitle("Property match vs training time and selected cache size")
    fig.text(0.5, 0.035,
             "Elapsed time is per diffusion run, excludes shared encoder/cache preparation, and reflects variable GPU contention.",
             ha="center", fontsize=8)
    fig.text(0.5, 0.012,
             "Codes: CB=ChemBERTa, LL=Llama, SR=small SMILES RoBERTa, F=frozen, S=scratch, P=token PCA, L=learned, MP=molecule PCA.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _save(fig, directory, "05_quality_cost_tradeoff")
    return costs


def _plot_scaffolds(metrics: pd.DataFrame, directory: Path) -> None:
    frame = metrics.sort_values("mean_normalized_w1").reset_index(drop=True)
    labels = [_condition_label(row) for row in frame.itertuples(index=False)]
    positions = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.55 * len(frame) + 1.3)),
                             sharey=True)
    for axis, column, title in zip(
        axes,
        ("scaffold_diversity_valid", "reference_scaffold_recall"),
        ("Distinct scaffolds per valid sample", "Reference scaffold recall"),
    ):
        axis.barh(positions, frame[column], color=[
            ENCODER_COLORS.get((row.encoder, row.variant), "#777777")
            for row in frame.itertuples(index=False)
        ])
        axis.set_title(title)
        axis.set_xlabel("Fraction")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Scaffold diversity and reference coverage")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, directory, "06_scaffold_metrics")


def _latent_sample(root: Path, data_seed: int, count: int) -> tuple[dict, np.ndarray, pd.DataFrame]:
    caches = {
        ("chemberta", "frozen"): root / "encoders/chemberta/frozen/shared/mean_pooled_raw/generative_embedding_cache.pt",
        ("chemberta", "scratch"): root / f"encoders/chemberta/scratch/seed_101/mean_pooled_raw/generative_embedding_cache.pt",
        ("llama", "frozen"): root / "encoders/llama/frozen/shared/mean_pooled_raw/generative_embedding_cache.pt",
    }
    small_cache = root / "encoders/smiles_roberta/scratch/seed_101/mean_pooled_raw/generative_embedding_cache.pt"
    if small_cache.exists():
        caches[("smiles_roberta", "scratch")] = small_cache
    selected = {}
    geometry_rows = []
    smiles_reference = None
    rng = np.random.default_rng(data_seed + 81)
    for key in (key for key in ENCODER_ORDER if key in caches):
        payload = torch.load(caches[key], map_location="cpu", weights_only=False, mmap=True)
        split = payload["splits"]["train"]
        if smiles_reference is None:
            indices = rng.permutation(len(split["smiles"]))[:count]
            smiles_reference = [split["smiles"][int(index)] for index in indices]
        else:
            current = [split["smiles"][int(index)] for index in indices]
            if current != smiles_reference:
                raise ValueError(f"Pooled cache molecule order differs for {key}")
        vectors = split["embeddings"][indices].numpy().astype(np.float64)
        if not np.isfinite(vectors).all():
            raise ValueError(f"Non-finite pooled embeddings for {key}")
        subset = vectors[:min(512, len(vectors))]
        centered = subset - subset.mean(axis=0, keepdims=True)
        eigenvalues = np.linalg.eigvalsh(centered @ centered.T)
        eigenvalues = eigenvalues[eigenvalues > eigenvalues.max() * 1e-12][::-1]
        weights = eigenvalues / eigenvalues.sum()
        rms_spread = float(np.sqrt(np.mean(centered * centered)))
        geometry_rows.append({
            "encoder": key[0], "variant": key[1],
            "geometry_count": len(subset),
            "effective_rank": float(np.exp(-np.sum(weights * np.log(weights)))),
            "top_10_variance_fraction": float(weights[:10].sum()),
            "pooled_feature_rms_variation": rms_spread,
            "precision": "float64",
        })
        projection = PCA(n_components=3, svd_solver="randomized", random_state=data_seed)
        coordinates = projection.fit_transform(vectors)
        selected[key] = (coordinates, projection.explained_variance_ratio_, rms_spread)
        del payload
    properties = []
    for smiles in smiles_reference:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("Invalid molecule in pooled cache sample")
        properties.append([molecule_property(molecule, name) for name in PROPERTIES])
    return selected, np.asarray(properties, dtype=float), pd.DataFrame(geometry_rows)


def _plot_latent_2d(latents: dict, properties: np.ndarray, directory: Path) -> None:
    colored = ("qed", "logp", "sas")
    keys = [key for key in ENCODER_ORDER if key in latents]
    fig, axes = plt.subplots(3, len(keys), figsize=(4.7 * len(keys), 12), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.06, top=0.86,
                        wspace=0.35, hspace=0.46)
    for row, prop in enumerate(colored):
        index = PROPERTIES.index(prop)
        values = properties[:, index]
        vmin, vmax = np.quantile(values, [0.01, 0.99])
        for column, key in enumerate(keys):
            axis = axes[row, column]
            coords, variance, spread = latents[key]
            micro_scale = spread < 1e-5
            display = coords * 1e6 if micro_scale else coords
            unit = ", ×10⁻⁶ units" if micro_scale else ""
            points = axis.scatter(display[:, 0], display[:, 1], c=values,
                                  cmap="viridis", vmin=vmin, vmax=vmax,
                                  s=7, alpha=0.5, rasterized=True)
            axis.set_xlabel(f"PC1 ({variance[0]:.1%}{unit})")
            axis.set_ylabel(f"PC2 ({variance[1]:.1%}{unit})")
            axis.grid(alpha=0.15)
            if row == 0:
                axis.set_title(f"{_encoder_label(*key)}\nRMS feature spread: {spread:.2e}")
        box = axes[row, -1].get_position()
        color_axis = fig.add_axes([0.90, box.y0, 0.012, box.height])
        fig.colorbar(points, cax=color_axis, label=LABELS[prop])
    fig.suptitle("Pooled encoder spaces: PCA views of identical training molecules",
                 y=0.98)
    _save(fig, directory, "07_latent_pca_2d")


def _plot_latent_3d(latents: dict, properties: np.ndarray, directory: Path) -> None:
    keys = [key for key in ENCODER_ORDER if key in latents]
    fig = plt.figure(figsize=(5 * len(keys), 5.5))
    values = properties[:, PROPERTIES.index("qed")]
    vmin, vmax = np.quantile(values, [0.01, 0.99])
    order = np.argsort(values)
    for column, key in enumerate(keys):
        axis = fig.add_subplot(1, len(keys), column + 1, projection="3d")
        coords, variance, spread = latents[key]
        micro_scale = spread < 1e-5
        display = coords * 1e6 if micro_scale else coords
        unit = " ×10⁻⁶" if micro_scale else ""
        points = axis.scatter(display[order, 0], display[order, 1], display[order, 2],
                              c=values[order], cmap="viridis", vmin=vmin, vmax=vmax,
                              s=5, alpha=0.45, rasterized=True)
        axis.set_title(f"{_encoder_label(*key)}\nRMS spread: {spread:.2e}")
        axis.set_xlabel(f"PC1{unit} ({variance[0]:.1%})")
        axis.set_ylabel(f"PC2{unit} ({variance[1]:.1%})")
        axis.set_zlabel("")
        axis.text2D(0.04, 0.02, f"PC3{unit}: {variance[2]:.1%} variance",
                    transform=axis.transAxes, fontsize=8)
        axis.view_init(elev=22, azim=-62)
    color_axis = fig.add_axes([0.92, 0.22, 0.012, 0.55])
    fig.colorbar(points, cax=color_axis, label="QED")
    fig.suptitle("Three-dimensional PCA of pooled encoder embeddings", y=0.98)
    fig.text(0.5, 0.015,
             "The same molecules appear in each panel; PCA is fitted separately, so axes are not aligned across models.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.88, bottom=0.08, top=0.82, wspace=0.02)
    _save(fig, directory, "08_latent_pca_3d")


def _plot_latent_diagnostics(diagnostics: pd.DataFrame, geometry: pd.DataFrame,
                             directory: Path) -> None:
    frame = diagnostics.copy()
    frame = frame.drop(columns=["effective_rank", "top_10_variance_fraction",
                                "pooled_feature_rms_variation"],
                       errors="ignore").merge(geometry, on=["encoder", "variant"],
                                                 validate="one_to_one")
    frame["label"] = [_encoder_label(row.encoder, row.variant)
                      for row in frame.itertuples(index=False)]
    frame["order"] = [ENCODER_ORDER.index((row.encoder, row.variant))
                      for row in frame.itertuples(index=False)]
    frame = frame.sort_values("order")
    colors = [ENCODER_COLORS[(row.encoder, row.variant)]
              for row in frame.itertuples(index=False)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    positions = np.arange(len(frame))
    axes[0, 0].bar(positions, frame.pooled_feature_rms_variation, color=colors)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Absolute spread of pooled embeddings")
    axes[0, 0].set_ylabel("Centered feature RMS (log scale)")
    axes[0, 1].bar(positions, frame.top_10_variance_fraction, color=colors)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Variance in first 10 PCs")
    axes[0, 1].set_ylabel("Fraction")
    axes[1, 0].bar(positions - 0.18, frame.neighbor_property_distance,
                   width=0.36, color=colors, label="Latent nearest neighbors")
    axes[1, 0].bar(positions + 0.18, frame.random_property_distance,
                   width=0.36, color="#aaaaaa", label="Random molecules")
    axes[1, 0].set_title("Chemical property distance in neighborhoods")
    axes[1, 0].set_ylabel("Mean normalized property difference")
    axes[1, 0].legend(fontsize=8)
    probes = np.asarray([
        [json.loads(row.scaffold_val_probe_r2)[name] for name in PROPERTIES]
        for row in frame.itertuples(index=False)
    ])
    image = axes[1, 1].imshow(probes, cmap="coolwarm", vmin=-1, vmax=1,
                              aspect="auto")
    axes[1, 1].set_xticks(range(4), ["MW", "logP", "QED", "SA"])
    axes[1, 1].set_yticks(positions, frame.label)
    axes[1, 1].set_title("Scaffold validation property probe R²")
    for row in range(len(frame)):
        for column in range(4):
            axes[1, 1].text(column, row, f"{probes[row, column]:.2f}",
                            ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1, 1], shrink=0.75)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(positions, frame.label, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Intrinsic pooled-encoder geometry and chemical information")
    fig.text(0.5, 0.01,
             "Scratch variation is near float32 precision; its relative PCA spectrum can reflect quantization noise.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, directory, "09_latent_geometry_and_probes")


def _plot_information_tradeoff(metrics: pd.DataFrame, diagnostics: pd.DataFrame,
                               geometry: pd.DataFrame, directory: Path) -> None:
    probes = diagnostics[["encoder", "variant", "scaffold_val_probe_r2"]].copy()
    probes["mean_probe_r2"] = probes.scaffold_val_probe_r2.map(
        lambda value: float(np.mean(list(json.loads(value).values())))
    )
    frame = metrics.merge(
        geometry[["encoder", "variant", "pooled_feature_rms_variation"]],
        on=["encoder", "variant"], validate="many_to_one",
    ).merge(
        probes[["encoder", "variant", "mean_probe_r2"]],
        on=["encoder", "variant"], validate="many_to_one",
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    offsets = {"pca": (5, 8), "learned": (5, -10), "mean_pooled_pca": (5, 2)}
    for row in frame.itertuples(index=False):
        color = ENCODER_COLORS.get((row.encoder, row.variant), "#777777")
        marker = {"pca": "o", "learned": "s", "mean_pooled_pca": "D"}.get(
            row.interface, "x"
        )
        for axis, x in ((axes[0], row.pooled_feature_rms_variation),
                        (axes[1], row.mean_probe_r2)):
            axis.scatter(x, row.mean_normalized_w1, color=color, marker=marker,
                         s=90, edgecolors="black", linewidths=0.4, zorder=3)
            offset = offsets.get(row.interface, (5, 4))
            if row.variant != "scratch":
                offset = (-6, offset[1])
            axis.annotate(_short_label(row), (x, row.mean_normalized_w1),
                          xytext=offset, textcoords="offset points", fontsize=7,
                          ha="right" if row.variant != "scratch" else "left")
    axes[0].set_xscale("log")
    axes[0].set_xlim(frame.pooled_feature_rms_variation.min() / 1.5,
                     frame.pooled_feature_rms_variation.max() * 1.8)
    axes[0].set_xlabel("Pooled encoder feature spread (RMS, log scale)")
    axes[1].set_xlim(frame.mean_probe_r2.min() - 0.04,
                     frame.mean_probe_r2.max() + 0.08)
    axes[1].set_xlabel("Mean scaffold-validation property probe R²")
    for axis in axes:
        axis.set_ylabel("Mean normalized W1 (lower is better)")
        axis.grid(alpha=0.2)
    fig.suptitle("Property distribution match vs encoder-space information")
    fig.text(0.5, 0.01,
             "Each encoder state supplies several diffusion conditions. Scratch spread is near float32 precision; association is descriptive, not causal.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, directory, "10_property_match_vs_latent_information")


def _sync_geometry_diagnostics(root: Path, geometry: pd.DataFrame,
                               expected_train_count: int) -> None:
    """Correct geometry fields in old float32 diagnostics for the same sample."""
    for row in geometry.itertuples(index=False):
        tag = "shared" if row.variant == "frozen" else "seed_101"
        path = root / "encoders" / row.encoder / row.variant / tag / "latent_diagnostics.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["train_count"] != expected_train_count:
            raise ValueError(f"Diagnostic sample count differs from latent sample: {path}")
        record.update({
            "effective_rank": row.effective_rank,
            "top_10_variance_fraction": row.top_10_variance_fraction,
            "pooled_feature_rms_variation": row.pooled_feature_rms_variation,
            "geometry_numerical_precision": "float64",
        })
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    diagnostic_rows = []
    for path in sorted((root / "encoders").glob("*/*/*/latent_diagnostics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update({
            "encoder": path.parts[-4],
            "variant": path.parts[-3],
            "encoder_seed": path.parts[-2],
        })
        record["scaffold_val_probe_r2"] = json.dumps(record["scaffold_val_probe_r2"])
        diagnostic_rows.append(record)
    pd.DataFrame(diagnostic_rows).to_csv(root / "analysis" / "latent_diagnostics.csv",
                                         index=False)


def generate(args) -> None:
    root = Path(args.output_root)
    output = root / "analysis"
    metrics = pd.read_csv(output / "per_run_metrics.csv")
    records = _load_records(root, metrics)
    protocol = json.loads((output / "analysis_protocol.json").read_text())
    if protocol["num_runs"] != len(records) or protocol["data_seed"] != args.data_seed:
        raise ValueError("Analysis protocol is stale or uses another data seed.")
    reference = _reference_properties(args.dataset_path, args.data_seed,
                                      protocol["reference_size_per_half"])
    generated = _generated_properties(records, metrics)
    diagnostics = pd.read_csv(output / "latent_diagnostics.csv")
    latents, latent_properties, geometry = _latent_sample(
        root, args.data_seed, args.latent_sample_count
    )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="new_study_plots_", dir=output) as temp:
        directory = Path(temp)
        _plot_property_distributions(reference, generated, directory)
        _plot_error_heatmap(metrics, directory)
        _plot_joint_properties(reference, generated, metrics, directory)
        _plot_training_curves(records, directory)
        costs = _plot_cost_tradeoff(metrics, directory)
        _plot_scaffolds(metrics, directory)
        _plot_latent_2d(latents, latent_properties, directory)
        _plot_latent_3d(latents, latent_properties, directory)
        _plot_latent_diagnostics(diagnostics, geometry, directory)
        _plot_information_tradeoff(metrics, diagnostics, geometry, directory)
        figure_names = sorted(path.name for path in directory.glob("*.pdf"))
        if len(figure_names) != 10 or len(list(directory.glob("*.png"))) != 10:
            raise RuntimeError("The replacement figure set is incomplete.")
        for old in list(output.glob("*.pdf")) + list(output.glob("*.png")):
            old.unlink()
        for new in directory.iterdir():
            new.replace(output / new.name)
    costs.to_csv(output / "cost_metrics.csv", index=False)
    geometry.to_csv(output / "latent_geometry_sample.csv", index=False)
    if args.latent_sample_count == 2000:
        _sync_geometry_diagnostics(root, geometry, args.latent_sample_count)
    manifest = {
        "schema_version": 1,
        "run_count": len(records),
        "training_seeds": sorted({record["key"][3] for record in records}),
        "reference_size": len(reference),
        "latent_sample_count": args.latent_sample_count,
        "figures": figure_names,
        "notes": [
            "Latent PCA is fitted separately for each encoder on identical molecules.",
            "Scratch pooled embeddings have near-constant absolute values; PCA of their small residual variation can reflect float32 quantization noise.",
            "Training time excludes shared encoder preparation and reflects GPU contention.",
            "Only one training seed is available per condition; no error bars are shown.",
        ],
    }
    (output / "plots_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Replaced old plots with {len(figure_names)} figures for {len(records)} runs: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/latent_space_study_v1")
    parser.add_argument("--dataset-path", default="data/250k_zinc_clean.csv")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--latent-sample-count", type=int, default=2000)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
