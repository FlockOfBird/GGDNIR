from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import six
import numpy as np
import pandas as pd
import torch

from .properties import (
    SUPPORTED_PROPERTIES,
    molecule_property,
    normalize_property_name,
    property_distribution,
    summarize_property,
)
from .validation import validate_smiles
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

REFERENCE_METRIC_SAMPLE_SIZE = 5000

# molsets/MOSES still imports rdkit.six, which was removed from newer RDKit builds.
sys.modules.setdefault("rdkit.six", six)

# molsets/MOSES also uses DataFrame.append, removed in pandas 2.x.
if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda self, other, sort=False: pd.concat([self, other], sort=sort)

try:
    from moses.metrics.metrics import FCDMetric
    from moses.metrics.metrics import internal_diversity as molsets_internal_diversity
except ImportError:
    FCDMetric = None
    molsets_internal_diversity = None


def _scaffold(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)


def _valid_canonical_smiles(smiles_list: list[str]) -> list[str]:
    return [
        result.canonical_smiles
        for result in (validate_smiles(smiles) for smiles in smiles_list)
        if result.is_valid and result.canonical_smiles
    ]


def _sample_reference_smiles(smiles_list: list[str], sample_size: int = REFERENCE_METRIC_SAMPLE_SIZE) -> list[str]:
    if len(smiles_list) <= sample_size:
        return smiles_list
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(len(smiles_list), size=sample_size, replace=False))
    return [smiles_list[index] for index in indices]


def _molecules(smiles_list: list[str]) -> list[Chem.Mol]:
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    return [molecule for molecule in molecules if molecule is not None]


def _descriptor_vector(molecule: Chem.Mol) -> list[float]:
    return [
        Descriptors.MolWt(molecule),
        Crippen.MolLogP(molecule),
        rdMolDescriptors.CalcTPSA(molecule),
        float(Lipinski.NumHAcceptors(molecule)),
        float(Lipinski.NumHDonors(molecule)),
        float(Lipinski.NumRotatableBonds(molecule)),
        QED.qed(molecule),
        molecule_property(molecule, "sas"),
    ]


def _descriptor_matrix(smiles_list: list[str]) -> np.ndarray:
    molecules = _molecules(smiles_list)
    if not molecules:
        return np.empty((0, 8), dtype=float)
    return np.asarray([_descriptor_vector(molecule) for molecule in molecules], dtype=float)


def _kl_divergence(generated: np.ndarray, reference: np.ndarray, bins: int = 50) -> dict[str, float] | None:
    if generated.size == 0 or reference.size == 0:
        return None
    descriptor_names = [
        "molecular_weight",
        "logp",
        "tpsa",
        "h_acceptors",
        "h_donors",
        "rotatable_bonds",
        "qed",
        "sas",
    ]
    eps = 1e-8
    scores = {}
    for index, name in enumerate(descriptor_names):
        combined = np.concatenate([generated[:, index], reference[:, index]])
        lower = float(np.min(combined))
        upper = float(np.max(combined))
        if lower == upper:
            scores[name] = 0.0
            continue
        gen_hist, bin_edges = np.histogram(generated[:, index], bins=bins, range=(lower, upper), density=False)
        ref_hist, _ = np.histogram(reference[:, index], bins=bin_edges, density=False)
        gen_dist = (gen_hist.astype(float) + eps) / (float(np.sum(gen_hist)) + eps * bins)
        ref_dist = (ref_hist.astype(float) + eps) / (float(np.sum(ref_hist)) + eps * bins)
        scores[name] = float(np.sum(gen_dist * np.log(gen_dist / ref_dist)))
    scores["mean"] = float(np.mean(list(scores.values())))
    return scores


def _rbf_kernel(values: np.ndarray, gamma: float) -> np.ndarray:
    squared_norms = np.sum(values * values, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * values @ values.T
    return np.exp(-gamma * np.maximum(distances, 0.0))


def _rbf_cross_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    left_norms = np.sum(left * left, axis=1, keepdims=True)
    right_norms = np.sum(right * right, axis=1, keepdims=True).T
    distances = left_norms + right_norms - 2.0 * left @ right.T
    return np.exp(-gamma * np.maximum(distances, 0.0))


def _mmd(generated: np.ndarray, reference: np.ndarray) -> float | None:
    if generated.size == 0 or reference.size == 0:
        return None
    combined = np.vstack([generated, reference])
    scale = np.std(combined, axis=0)
    scale[scale == 0.0] = 1.0
    generated = (generated - np.mean(combined, axis=0)) / scale
    reference = (reference - np.mean(combined, axis=0)) / scale
    gamma = 1.0 / generated.shape[1]
    gen_kernel = _rbf_kernel(generated, gamma)
    ref_kernel = _rbf_kernel(reference, gamma)
    cross_kernel = _rbf_cross_kernel(generated, reference, gamma)
    return float(gen_kernel.mean() + ref_kernel.mean() - 2.0 * cross_kernel.mean())


def _frechet_chemnet_distance(generated_smiles: list[str], reference_smiles: list[str], device: torch.device) -> float | None:
    if FCDMetric is None or not generated_smiles or not reference_smiles:
        return None
    reference_statistics = FCDMetric(n_jobs=1, device=str(device)).precalc(reference_smiles)
    return float(FCDMetric(n_jobs=1, device=str(device))(gen=generated_smiles, pref=reference_statistics))


def _internal_diversity(smiles_list: list[str], device: torch.device) -> float | None:
    if molsets_internal_diversity is None:
        return None
    molecules = _molecules(smiles_list)
    if not molecules:
        return 0.0
    return float(molsets_internal_diversity(molecules, n_jobs=1, device=str(device)))


def generation_metrics(
    smiles_list: list[str],
    training_smiles: list[str],
    property_name: str,
    guidance_scale: float,
    device: torch.device,
) -> dict[str, float | int | None | list | dict]:
    property_name = normalize_property_name(property_name)
    total = len(smiles_list)
    validation_results = [validate_smiles(smiles) for smiles in smiles_list]
    valid_smiles = [result.canonical_smiles for result in validation_results if result.is_valid and result.canonical_smiles]
    unique_valid_smiles = set(valid_smiles)
    reference_smiles = _valid_canonical_smiles(training_smiles)
    sampled_reference_smiles = _sample_reference_smiles(reference_smiles)
    training_set = set(reference_smiles)
    novel_valid_count = sum(smiles not in training_set for smiles in valid_smiles)

    generated_descriptors = _descriptor_matrix(valid_smiles)
    reference_descriptors = _descriptor_matrix(sampled_reference_smiles)

    generated_molecules = _molecules(valid_smiles)
    reference_molecules = _molecules(sampled_reference_smiles)
    generated_property_values = {
        name: [molecule_property(molecule, name) for molecule in generated_molecules]
        for name in SUPPORTED_PROPERTIES
    }
    reference_property_values_by_name = {
        name: [molecule_property(molecule, name) for molecule in reference_molecules]
        for name in SUPPORTED_PROPERTIES
    }
    property_distributions = {
        name: property_distribution(
            generated_property_values[name], reference_property_values_by_name[name]
        )
        for name in SUPPORTED_PROPERTIES
    }
    property_values = generated_property_values[property_name]
    reference_property_values = reference_property_values_by_name[property_name]
    property_summary = summarize_property(property_values)
    reference_property_summary = summarize_property(reference_property_values)
    property_shift = None
    direction_aligned_shift = None
    if property_values and reference_property_values:
        property_shift = float(property_summary["mean"] - reference_property_summary["mean"])
        if guidance_scale != 0.0:
            direction_aligned_shift = (
                property_shift if guidance_scale > 0.0 else -property_shift
            )

    scaffolds = [_scaffold(smiles) for smiles in valid_smiles]
    scaffolds = [scaffold for scaffold in scaffolds if scaffold]
    scaffold_diversity = len(set(scaffolds)) / len(scaffolds) if scaffolds else 0.0

    return {
        "num_samples": total,
        "validity": len(valid_smiles) / max(total, 1),
        "uniqueness": len(unique_valid_smiles) / max(total, 1),
        "novelty": novel_valid_count / max(total, 1),
        "reference_metric_sample_size": len(sampled_reference_smiles),
        "frechet_chemnet_distance": _frechet_chemnet_distance(valid_smiles, sampled_reference_smiles, device),
        "kl_divergence": _kl_divergence(generated_descriptors, reference_descriptors),
        "mmd": _mmd(generated_descriptors, reference_descriptors),
        "internal_diversity": _internal_diversity(valid_smiles, device),
        "property_name": property_name,
        "guidance_scale": guidance_scale,
        "guidance_direction": (
            "increase" if guidance_scale > 0.0 else "decrease" if guidance_scale < 0.0 else "unguided"
        ),
        "generated_property": property_summary,
        "reference_property": reference_property_summary,
        "property_distribution": property_distributions[property_name],
        "property_distributions": property_distributions,
        "property_mean_shift": property_shift,
        "direction_aligned_property_shift": direction_aligned_shift,
        "binding_affinity_docking_scores": None,
        "reconstruction_accuracy": None,
        "scaffold_diversity": scaffold_diversity,
        "top_scaffolds": Counter(scaffolds).most_common(5),
    }

_PROPERTY_PLOT_LABELS = {
    "logp": "logP",
    "sas": "Synthetic accessibility score",
    "qed": "QED",
    "molecular_weight": "Molecular weight",
}


def _draw_property_distribution(
    axis,
    property_name: str,
    distribution: dict,
) -> None:
    edges = np.asarray(distribution.get("bin_edges", []), dtype=float)
    generated = np.asarray(
        distribution.get("generated_probabilities", []), dtype=float
    )
    reference = np.asarray(
        distribution.get("reference_probabilities", []), dtype=float
    )
    label = _PROPERTY_PLOT_LABELS.get(property_name, property_name)

    if edges.size < 2:
        axis.text(
            0.5,
            0.5,
            "No valid property values",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        if edges.size != generated.size + 1 or edges.size != reference.size + 1:
            raise ValueError(
                f"Invalid shared-bin distribution for {property_name!r}."
            )
        axis.stairs(
            reference,
            edges,
            label="Reference ZINC",
            color="#4C78A8",
            linewidth=2.0,
        )
        axis.stairs(
            generated,
            edges,
            label="Generated",
            color="#E45756",
            linewidth=2.0,
        )
        axis.legend()
    axis.set_title(label)
    axis.set_xlabel(label)
    axis.set_ylabel("Probability mass")
    axis.grid(alpha=0.2)


def save_property_distribution_plots(
    property_distributions: dict[str, dict],
    output_dir: str | Path,
) -> dict[str, str]:
    """Save one combined generated/reference property plot."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    property_names = [
        name for name in SUPPORTED_PROPERTIES if name in property_distributions
    ]
    if not property_names:
        raise ValueError("No supported property distributions were provided.")

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, property_name in zip(axes.flat, property_names):
        _draw_property_distribution(
            axis,
            property_name,
            property_distributions[property_name],
        )
    for axis in list(axes.flat)[len(property_names) :]:
        axis.set_visible(False)
    figure.suptitle("Generated and reference property distributions")
    figure.tight_layout()
    combined_path = output_dir / "property_distributions.png"
    figure.savefig(combined_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return {"combined": str(combined_path)}
