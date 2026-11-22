"""Matched distribution evaluation for the latent-space study."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import wasserstein_distance

from .config import DataConfig
from .data import build_data_bundle
from .properties import SUPPORTED_PROPERTIES, molecule_property

RDLogger.DisableLog("rdApp.error")
PROPERTIES = ("molecular_weight", "logp", "qed", "sas")
LABELS = {
    "molecular_weight": "Molecular weight",
    "logp": "logP",
    "qed": "QED",
    "sas": "Synthetic accessibility",
}
CONDITION_STYLES = {
    ("frozen", "pca"): ("Frozen token PCA", "#0072B2", "-"),
    ("scratch", "pca"): ("Scratch token PCA", "#0072B2", "--"),
    ("frozen", "learned"): ("Frozen learned projection", "#D55E00", "-"),
    ("scratch", "learned"): ("Scratch learned projection", "#D55E00", "--"),
    ("frozen", "mean_pooled_pca"): ("Frozen mean-pooled PCA", "#009E73", "-"),
    ("scratch", "mean_pooled_pca"): ("Scratch mean-pooled PCA", "#009E73", "--"),
}


def describe_smiles(smiles: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    values = []
    canonical = []
    scaffolds = []
    for item in smiles:
        molecule = Chem.MolFromSmiles(str(item)) if item else None
        if molecule is None:
            continue
        canonical.append(Chem.MolToSmiles(molecule, canonical=True))
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=molecule))
        values.append([molecule_property(molecule, name) for name in PROPERTIES])
    matrix = np.asarray(values, dtype=np.float64).reshape(-1, len(PROPERTIES))
    return matrix, canonical, scaffolds


def _sliced_wasserstein(left: np.ndarray, right: np.ndarray, scales: np.ndarray) -> float:
    rng = np.random.default_rng(1245)
    directions = rng.normal(size=(64, left.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    left_scaled = left / scales
    right_scaled = right / scales
    return float(np.mean([
        wasserstein_distance(left_scaled @ direction, right_scaled @ direction)
        for direction in directions
    ]))


def distribution_scores(generated: np.ndarray, reference: np.ndarray,
                        scales: np.ndarray) -> dict[str, float]:
    if len(generated) == 0 or len(reference) == 0:
        raise ValueError("Both generated and reference property matrices must be nonempty.")
    scores = {
        f"w1_{name}": float(
            wasserstein_distance(generated[:, index], reference[:, index]) / scales[index]
        )
        for index, name in enumerate(PROPERTIES)
    }
    scores["mean_normalized_w1"] = float(np.mean(list(scores.values())))
    scores["joint_sliced_w1"] = _sliced_wasserstein(generated, reference, scales)
    return scores


def evaluate_condition(sample_path: Path, *, train_smiles: set[str],
                       train_reference: np.ndarray, train_floor: np.ndarray,
                       test_reference: np.ndarray, scales: np.ndarray,
                       reference_scaffolds: set[str]) -> tuple[dict, np.ndarray]:
    samples = pd.read_csv(sample_path)
    if "SMILES" not in samples:
        raise ValueError(f"Missing SMILES column: {sample_path}")
    attempted = samples["SMILES"].fillna("").astype(str).tolist()
    generated, canonical, scaffolds = describe_smiles(attempted)
    if len(generated) == 0:
        raise ValueError(f"No valid generated molecules in {sample_path}")
    n = min(len(generated), len(train_reference), len(train_floor), len(test_reference))
    generated = generated[:n]
    train = train_reference[:n]
    floor = train_floor[:n]
    test = test_reference[:n]
    generated_scaffolds = {value for value in scaffolds if value}
    row = {
        "attempted": len(attempted),
        "valid": len(canonical),
        "property_score_count": n,
        "primary_metric_eligible": n >= 1000,
        "validity": len(canonical) / max(len(attempted), 1),
        "uniqueness_valid": len(set(canonical)) / len(canonical),
        "novelty_valid": sum(value not in train_smiles for value in canonical) / len(canonical),
        "scaffold_diversity_valid": len(generated_scaffolds) / len(canonical),
        "reference_scaffold_recall": (
            len(generated_scaffolds & reference_scaffolds) / len(reference_scaffolds)
            if reference_scaffolds else 0.0
        ),
    }
    row.update(distribution_scores(generated, train, scales))
    row.update({f"test_{key}": value for key, value in distribution_scores(generated, test, scales).items()})
    row.update({f"floor_{key}": value for key, value in distribution_scores(floor, train, scales).items()})
    row["excess_mean_normalized_w1"] = max(
        0.0, row["mean_normalized_w1"] - row["floor_mean_normalized_w1"]
    )
    return row, generated


def _plot_distributions(reference: np.ndarray, groups: dict[tuple[str, str], list[np.ndarray]],
                        output_dir: Path) -> None:
    interfaces = sorted({key[1] for key in groups})
    for interface in interfaces:
        selected = sorted(
            (key, matrices) for key, matrices in groups.items() if key[1] == interface
        )
        if not selected:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected), 1)))
        for index, name in enumerate(PROPERTIES):
            axis = axes.flat[index]
            ref_values = reference[:, index]
            lower, upper = np.quantile(ref_values, [0.005, 0.995])
            span = max(upper - lower, 1e-6)
            bins = np.linspace(lower - 0.1 * span, upper + 0.1 * span, 36)
            axis.hist(ref_values, bins=bins,
                      weights=np.full(len(ref_values), 1.0 / len(ref_values)),
                      histtype="step", linewidth=2.4,
                      color="black", label="Reference train")
            for color, ((label, _), matrices) in zip(colors, selected):
                values = np.concatenate(matrices)[:, index]
                axis.hist(values, bins=bins,
                          weights=np.full(len(values), 1.0 / len(values)),
                          histtype="step", linewidth=1.5, color=color,
                          label=label.replace("_", " "))
            axis.set_title(LABELS[name])
            axis.set_xlabel(LABELS[name])
            axis.set_ylabel("Probability per bin")
            axis.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
        title = interface.replace("_", " ").upper()
        fig.suptitle(f"Generated and reference property distributions: {title}")
        fig.text(0.5, 0.015,
                 "Display range follows reference 0.5–99.5% quantiles; scores use all valid molecules.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.09, 1, 0.96))
        fig.savefig(output_dir / f"property_distributions_{interface}.pdf")
        fig.savefig(output_dir / f"property_distributions_{interface}.png", dpi=180)
        plt.close(fig)


def _plot_all_distributions(reference: np.ndarray,
                            groups: dict[tuple[str, str], list[np.ndarray]],
                            output_dir: Path) -> None:
    """Show every completed condition, with shared scales for encoder comparisons."""
    encoders = sorted({label.rsplit("_", 1)[0] for label, _ in groups})
    fig, axes = plt.subplots(
        len(PROPERTIES), len(encoders),
        figsize=(7.2 * len(encoders), 3.0 * len(PROPERTIES) + 1.8),
        sharex="row", sharey="row", squeeze=False,
    )
    condition_order = {condition: index for index, condition in enumerate(CONDITION_STYLES)}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0].rsplit("_", 1)[0],
            condition_order.get((item[0][0].rsplit("_", 1)[1], item[0][1]),
                                len(condition_order)),
        ),
    )
    present_styles = set()
    for property_index, name in enumerate(PROPERTIES):
        ref_values = reference[:, property_index]
        lower, upper = np.quantile(ref_values, [0.005, 0.995])
        span = max(upper - lower, 1e-6)
        bins = np.linspace(lower - 0.1 * span, upper + 0.1 * span, 36)
        for encoder_index, encoder in enumerate(encoders):
            axis = axes[property_index, encoder_index]
            axis.hist(
                ref_values, bins=bins,
                weights=np.full(len(ref_values), 1.0 / len(ref_values)),
                histtype="step", linewidth=2.5, color="black",
            )
            for (label, interface), matrices in ordered_groups:
                group_encoder, variant = label.rsplit("_", 1)
                if group_encoder != encoder:
                    continue
                style = CONDITION_STYLES.get(
                    (variant, interface),
                    (f"{variant} {interface}", "#777777", ":"),
                )
                present_styles.add((variant, interface))
                values = np.concatenate(matrices)[:, property_index]
                axis.hist(
                    values, bins=bins,
                    weights=np.full(len(values), 1.0 / len(values)),
                    histtype="step", linewidth=1.7,
                    color=style[1], linestyle=style[2],
                )
            axis.set_xlim(bins[0], bins[-1])
            axis.grid(alpha=0.2)
            if encoder_index == 0:
                axis.set_ylabel(f"{LABELS[name]}\nProbability per bin")
            if property_index == len(PROPERTIES) - 1:
                axis.set_xlabel(LABELS[name])
            if property_index == 0:
                axis.set_title(encoder.capitalize())
    handles = [Line2D([0], [0], color="black", linewidth=2.5, label="Reference train")]
    for condition, (label, color, linestyle) in CONDITION_STYLES.items():
        if condition in present_styles:
            handles.append(Line2D(
                [0], [0], color=color, linestyle=linestyle,
                linewidth=1.8, label=label,
            ))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Generated vs reference property distributions: all completed runs")
    fig.text(
        0.5, 0.012,
        "Shared bins and axes by property. Display range follows reference 0.5–99.5% quantiles; scores use all valid molecules.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.965))
    fig.savefig(output_dir / "property_distributions_all.pdf")
    fig.savefig(output_dir / "property_distributions_all.png", dpi=180)
    plt.close(fig)


def _write_comparison_table(per_run: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        "encoder", "variant", "latent_representation", "interface", "training_seed",
        "attempted", "valid", "validity", "uniqueness_valid", "novelty_valid",
        "w1_molecular_weight", "w1_logp", "w1_qed", "w1_sas",
        "mean_normalized_w1", "joint_sliced_w1", "test_mean_normalized_w1",
        "floor_mean_normalized_w1",
    ]
    table = per_run[columns].sort_values("mean_normalized_w1").reset_index(drop=True)
    table.to_csv(output_dir / "comparison_table.csv", index=False)
    lines = [
        "# Completed-run comparison",
        "",
        "One training seed per condition. Distances compare generated molecules with the same training reference; lower is better. "
        "Mean-pooled PCA uses one vector per molecule and changes the diffusion input shape.",
        "",
        "| Encoder | State | Interface | Valid / attempted | Mean W1 | MW W1 | logP W1 | QED W1 | SA W1 | Joint W1 | Unique | Novel |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.encoder.capitalize()} | {row.variant} | {row.interface} | "
            f"{row.valid} / {row.attempted} | {row.mean_normalized_w1:.3f} | "
            f"{row.w1_molecular_weight:.3f} | {row.w1_logp:.3f} | "
            f"{row.w1_qed:.3f} | {row.w1_sas:.3f} | {row.joint_sliced_w1:.3f} | "
            f"{row.uniqueness_valid:.3f} | {row.novelty_valid:.3f} |"
        )
    lines.extend([
        "",
        "W1 values are normalized by the reference interquartile range. Unique and novel fractions use valid molecules as the denominator. "
        "See `comparison_table.csv` for validity, scaffold-test distance, and full-precision values.",
        "",
    ])
    (output_dir / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_primary_scores(per_run: pd.DataFrame, output_dir: Path) -> None:
    grouped = per_run.groupby(
        ["encoder", "variant", "interface"], sort=True
    )
    summary = grouped["mean_normalized_w1"].agg(["mean", "std", "count"]).reset_index()
    summary["label"] = (
        summary["encoder"] + " / " + summary["variant"] + " / " + summary["interface"]
    )
    summary = summary.sort_values(["interface", "variant", "encoder"])
    fig, axis = plt.subplots(figsize=(10, max(4, 0.48 * len(summary) + 1.5)))
    positions = np.arange(len(summary))
    colors = ["#4C78A8" if value == "chemberta" else "#E45756"
              for value in summary["encoder"]]
    axis.barh(positions, summary["mean"], color=colors, alpha=0.7)
    axis.errorbar(summary["mean"], positions, xerr=summary["std"].fillna(0),
                  fmt="none", ecolor="black", capsize=3)
    axis.set_yticks(positions, summary["label"])
    axis.invert_yaxis()
    axis.set_xlabel("Mean normalized 1-Wasserstein distance (lower is better)")
    axis.set_title("Property-distribution match across training seeds")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "primary_score_summary.pdf")
    fig.savefig(output_dir / "primary_score_summary.png", dpi=180)
    plt.close(fig)


def paired_scratch_encoder_differences(per_run: pd.DataFrame) -> pd.DataFrame:
    """Compare the small and original scratch models within seed and interface."""
    rows = []
    for (interface, seed), frame in per_run.loc[per_run.variant == "scratch"].groupby(
        ["interface", "training_seed"]
    ):
        if not {"chemberta", "smiles_roberta"}.issubset(set(frame.encoder)):
            continue
        baseline = frame.loc[frame.encoder == "chemberta"].iloc[0]
        small = frame.loc[frame.encoder == "smiles_roberta"].iloc[0]
        rows.append({
            "interface": interface, "training_seed": seed,
            **{f"small_minus_existing_{metric}": small[metric] - baseline[metric]
               for metric in ("mean_normalized_w1", "joint_sliced_w1", "validity")},
        })
    return pd.DataFrame(rows, columns=[
        "interface", "training_seed", "small_minus_existing_mean_normalized_w1",
        "small_minus_existing_joint_sliced_w1", "small_minus_existing_validity",
    ])


def evaluate_study(args) -> None:
    root = Path(args.output_root)
    manifests = sorted((root / "runs").glob("*/*/*/seed_*/condition.json"))
    if not manifests:
        raise FileNotFoundError(f"No completed study runs under {root / 'runs'}")
    data_config = DataConfig(
        dataset_path=args.dataset_path, label_column=None, property_name=None,
        seed=args.data_seed, split_strategy="scaffold",
    )
    splits = build_data_bundle(data_config)
    train_smiles = splits.train[data_config.smiles_column].astype(str).tolist()
    test_smiles = splits.test[data_config.smiles_column].astype(str).tolist()
    if len(train_smiles) < 2 * args.reference_size or len(test_smiles) < args.reference_size:
        raise ValueError("Reference size exceeds the available train/test split size.")
    rng = np.random.default_rng(args.data_seed + 2026)
    train_indices = rng.permutation(len(train_smiles))[:2 * args.reference_size]
    test_indices = rng.permutation(len(test_smiles))[:args.reference_size]
    train_a, _, train_scaffolds = describe_smiles([
        train_smiles[index] for index in train_indices[:args.reference_size]
    ])
    train_b, _, _ = describe_smiles([
        train_smiles[index] for index in train_indices[args.reference_size:]
    ])
    test, _, _ = describe_smiles([test_smiles[index] for index in test_indices])
    scales = np.subtract(*np.quantile(train_a, [0.75, 0.25], axis=0))
    if np.any(scales <= 1e-8):
        raise ValueError("Reference IQR contains a near-zero scale.")
    rows = []
    training_set = set(train_smiles)
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["data_seed"] != args.data_seed:
            raise ValueError(f"Data seed mismatch: {manifest_path}")
        sample_path = Path(manifest["sample_path"])
        if not sample_path.is_absolute():
            sample_path = Path.cwd() / sample_path
        result, _ = evaluate_condition(
            sample_path, train_smiles=training_set,
            train_reference=train_a, train_floor=train_b,
            test_reference=test, scales=scales,
            reference_scaffolds={value for value in train_scaffolds if value},
        )
        latent_representation = manifest.get("latent_representation", "token_sequence")
        interface = manifest.get("condition_projection") or (
            f"mean_pooled_{manifest['projection']}"
            if latent_representation in ("mean_pooled", "molecule_mean_pooled_v1")
            else manifest["projection"]
        )
        row = {key: manifest[key] for key in (
            "encoder", "variant", "projection", "training_seed", "sampling_seed",
            "requested_samples",
        )}
        row["latent_representation"] = latent_representation
        row["interface"] = interface
        row.update({key: manifest.get(key) for key in (
            "train_seconds_this_invocation", "sampling_seconds_this_invocation",
            "encoder_adaptation_seconds", "raw_cache_seconds", "pooled_cache_seconds",
            "pca_seconds",
            "raw_cache_bytes", "selected_cache_bytes",
        )})
        row.update(result)
        rows.append(row)
    output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run = pd.DataFrame(rows).sort_values(
        ["interface", "variant", "encoder", "training_seed"]
    )
    per_run.to_csv(output_dir / "per_run_metrics.csv", index=False)
    _write_comparison_table(per_run, output_dir)
    metric_columns = [
        column for column in per_run.select_dtypes(include="number")
        if column not in ("training_seed", "sampling_seed", "requested_samples")
    ]
    aggregate = per_run.groupby(
        ["encoder", "variant", "projection", "latent_representation", "interface"],
        as_index=False,
    )[
        metric_columns
    ].agg(["mean", "std", "count"])
    aggregate.columns = ["_".join(item).strip("_") if isinstance(item, tuple) else item
                         for item in aggregate.columns]
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    paired_rows = []
    for (variant, projection, latent_representation, interface, seed), frame in per_run.groupby(
        [
            "variant",
            "projection",
            "latent_representation",
            "interface",
            "training_seed",
        ]
    ):
        if set(frame["encoder"]) != {"chemberta", "llama"}:
            continue
        chem = frame.loc[frame["encoder"] == "chemberta"].iloc[0]
        llama = frame.loc[frame["encoder"] == "llama"].iloc[0]
        paired_rows.append({
            "variant": variant,
            "projection": projection,
            "latent_representation": latent_representation,
            "interface": interface,
            "training_seed": seed,
            "chemberta_minus_llama_mean_normalized_w1": (
                chem["mean_normalized_w1"] - llama["mean_normalized_w1"]
            ),
            "chemberta_minus_llama_validity": chem["validity"] - llama["validity"],
            "chemberta_minus_llama_joint_sliced_w1": (
                chem["joint_sliced_w1"] - llama["joint_sliced_w1"]
            ),
        })
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_encoder_differences.csv", index=False)
    paired_scratch_encoder_differences(per_run).to_csv(
        output_dir / "paired_scratch_encoder_differences.csv", index=False,
    )
    # Figures are generated separately by study_plots after all metrics are written.
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
    if diagnostic_rows:
        pd.DataFrame(diagnostic_rows).to_csv(output_dir / "latent_diagnostics.csv", index=False)
    metadata = {
        "schema_version": 1,
        "primary_metric": "mean_normalized_w1",
        "primary_metric_direction": "lower_is_better",
        "properties": list(PROPERTIES),
        "reference_split": "train",
        "generalization_split": "scaffold_test",
        "data_seed": args.data_seed,
        "reference_size_per_half": args.reference_size,
        "reference_scale": "train_reference_IQR",
        "reference_scales": dict(zip(PROPERTIES, scales.tolist())),
        "num_runs": len(rows),
        "minimum_valid_count": min(row["valid"] for row in rows),
    }
    (output_dir / "analysis_protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Evaluated {len(rows)} runs; output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/latent_space_study_v1")
    parser.add_argument("--dataset-path", default="data/250k_zinc_clean.csv")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--reference-size", type=int, default=5000)
    args = parser.parse_args()
    evaluate_study(args)


if __name__ == "__main__":
    main()
