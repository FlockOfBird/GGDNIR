from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED


PROPERTY_COLUMNS = {
    "logp": "logP",
    "sas": "SAS",
    "qed": "qed",
    "molecular_weight": "molecular_weight",
}
SUPPORTED_PROPERTIES = tuple(PROPERTY_COLUMNS)


def normalize_property_name(name: str) -> str:
    normalized = str(name).strip().lower()
    aliases = {
        "log_p": "logp",
        "sa": "sas",
        "synthetic_accessibility": "sas",
        "mw": "molecular_weight",
        "molwt": "molecular_weight",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PROPERTY_COLUMNS:
        raise ValueError(
            f"Unsupported property {name!r}; choose from {', '.join(SUPPORTED_PROPERTIES)}."
        )
    return normalized


def property_column(name: str) -> str:
    return PROPERTY_COLUMNS[normalize_property_name(name)]


def _sas_score(molecule: Chem.Mol) -> float:
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError as error:
        raise RuntimeError(
            "SAS evaluation requires RDKit's Contrib/SA_Score/sascorer module."
        ) from error
    return float(sascorer.calculateScore(molecule))


def molecule_property(molecule: Chem.Mol, property_name: str) -> float:
    property_name = normalize_property_name(property_name)
    if property_name == "logp":
        return float(Crippen.MolLogP(molecule))
    if property_name == "sas":
        return _sas_score(molecule)
    if property_name == "qed":
        return float(QED.qed(molecule))
    return float(Descriptors.MolWt(molecule))


def smiles_property(smiles: str, property_name: str) -> float | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return molecule_property(molecule, property_name)


def summarize_property(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def property_distribution(
    generated_values: Iterable[float],
    reference_values: Iterable[float],
    num_bins: int = 20,
) -> dict[str, int | list[float] | list[int]]:
    """Build shared-bin generated/reference property histograms."""
    if num_bins < 1:
        raise ValueError("num_bins must be positive.")
    generated = np.asarray(list(generated_values), dtype=np.float64)
    reference = np.asarray(list(reference_values), dtype=np.float64)
    nonempty = [values for values in (generated, reference) if values.size]
    if not nonempty:
        return {
            "num_bins": num_bins,
            "bin_edges": [],
            "generated_counts": [],
            "generated_probabilities": [],
            "reference_counts": [],
            "reference_probabilities": [],
        }

    combined = np.concatenate(nonempty)
    lower = float(combined.min())
    upper = float(combined.max())
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1e-6)
        lower -= padding
        upper += padding
    bin_edges = np.linspace(lower, upper, num_bins + 1, dtype=np.float64)
    generated_counts = np.histogram(generated, bins=bin_edges)[0]
    reference_counts = np.histogram(reference, bins=bin_edges)[0]

    def probabilities(counts: np.ndarray) -> list[float]:
        total = int(counts.sum())
        if total == 0:
            return [0.0] * len(counts)
        return (counts.astype(np.float64) / total).tolist()

    return {
        "num_bins": num_bins,
        "bin_edges": bin_edges.tolist(),
        "generated_counts": generated_counts.astype(int).tolist(),
        "generated_probabilities": probabilities(generated_counts),
        "reference_counts": reference_counts.astype(int).tolist(),
        "reference_probabilities": probabilities(reference_counts),
    }
