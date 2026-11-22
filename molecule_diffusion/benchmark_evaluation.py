"""Evaluate saved samples with the official GuacaMol or MOSES benchmark suites."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
import six
from rdkit import Chem


def _samples(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "SMILES" not in frame:
        raise ValueError(f"Sample CSV has no SMILES column: {path}")
    return frame["SMILES"].fillna("").astype(str).tolist()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unique_valid(smiles: list[str]) -> list[str]:
    unique: dict[str, None] = {}
    for item in smiles:
        molecule = Chem.MolFromSmiles(item) if item else None
        if molecule is not None:
            # GuacaMol's goal benchmark deduplicates without stereocenters.
            unique[Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)] = None
    return list(unique)


def _rank_candidates(
    candidates: list[str], scoring_function, number_molecules: int,
    starting_population: list[str] | None = None,
) -> list[str]:
    pool = (
        _unique_valid([*starting_population, *candidates])
        if starting_population else candidates
    )
    return sorted(
        pool, key=lambda item: float(scoring_function.score(item)), reverse=True
    )[:number_molecules]


def evaluate_guacamol(
    samples: list[str], *, train_file: Path, output_dir: Path, goal_directed: bool,
) -> dict:
    # GuacaMol 0.5.5 imports scipy.histogram, removed from recent SciPy.
    if not hasattr(scipy, "histogram"):
        scipy.histogram = np.histogram
    try:
        from guacamol.assess_distribution_learning import assess_distribution_learning
        from guacamol.distribution_matching_generator import DistributionMatchingGenerator
        from guacamol.frechet_benchmark import FrechetBenchmark
    except ImportError as error:
        raise RuntimeError("Install requirements-benchmarks.txt to evaluate GuacaMol.") from error
    if len(samples) < 10_000:
        raise ValueError("Official GuacaMol distribution evaluation needs at least 10,000 attempts.")

    # GuacaMol 0.5.5 hardcodes the TensorFlow-era .h5 filename. FCD 1.2.x
    # packages the same ChemNet architecture as a PyTorch .pt checkpoint.
    original_frechet_init = FrechetBenchmark.__init__

    def current_frechet_init(self, training_set, chemnet_model_filename="ChemNet_v0.13_pretrained.pt", sample_size=10000):
        original_frechet_init(self, training_set, chemnet_model_filename, sample_size)

    FrechetBenchmark.__init__ = current_frechet_init

    class SavedSamples(DistributionMatchingGenerator):
        def __init__(self):
            self.cursor = 0

        def generate(self, number_samples: int) -> list[str]:
            result = [samples[(self.cursor + index) % len(samples)] for index in range(number_samples)]
            self.cursor = (self.cursor + number_samples) % len(samples)
            return result

    distribution_path = output_dir / "guacamol_distribution.json"
    assess_distribution_learning(
        SavedSamples(), chembl_training_file=str(train_file),
        json_output_file=str(distribution_path), benchmark_version="v2",
    )
    result = {
        "benchmark": "guacamol", "version": "v2", "attempts_available": len(samples),
        "distribution_result": str(distribution_path),
        "fcd_backend": "fcd_1.2_pytorch",
    }
    if goal_directed:
        from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
        from guacamol.goal_directed_generator import GoalDirectedGenerator

        candidates = _unique_valid(samples)
        if len(candidates) < 100:
            raise ValueError("Goal-directed sample-and-rank requires at least 100 distinct valid molecules.")

        class SampleAndRank(GoalDirectedGenerator):
            def generate_optimized_molecules(
                self, scoring_function, number_molecules: int,
                starting_population=None,
            ) -> list[str]:
                return _rank_candidates(
                    candidates, scoring_function, number_molecules, starting_population
                )

        goal_path = output_dir / "guacamol_goal_directed_sample_rank.json"
        assess_goal_directed_generation(
            SampleAndRank(), json_output_file=str(goal_path), benchmark_version="v2"
        )
        result.update({
            "goal_directed_method": "fixed_sample_pool_ranked_per_task",
            "goal_directed_candidate_count": len(candidates),
            "goal_directed_result": str(goal_path),
        })
    return result


def evaluate_moses(samples: list[str], *, output_dir: Path, device: str) -> dict:
    if len(samples) < 30_000:
        raise ValueError("Official MOSES evaluation recommends at least 30,000 attempts.")
    # MOSES 0.3.1 imports APIs removed from current RDKit and pandas.
    sys.modules.setdefault("rdkit.six", six)
    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = lambda self, other, sort=False: pd.concat([self, other], sort=sort)
    try:
        import moses
    except ImportError as error:
        raise RuntimeError("Install requirements-benchmarks.txt to evaluate MOSES.") from error
    metrics = _jsonable(moses.get_all_metrics(samples, n_jobs=1, device=device))
    output_path = output_dir / "moses_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {
        "benchmark": "moses", "attempts_available": len(samples),
        "official_train_test_and_scaffold_references": True,
        "metrics_file": str(output_path), "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["guacamol", "moses"], required=True)
    parser.add_argument("--samples-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--guacamol-train-file", type=Path)
    parser.add_argument("--goal-directed", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.benchmark != "guacamol" and args.goal_directed:
        parser.error("--goal-directed is a GuacaMol-only option.")
    if args.benchmark == "guacamol" and (
        args.guacamol_train_file is None or not args.guacamol_train_file.is_file()
    ):
        parser.error("GuacaMol requires --guacamol-train-file pointing to the official train SMILES.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = _samples(args.samples_csv)
    result = (
        evaluate_guacamol(
            samples, train_file=args.guacamol_train_file,
            output_dir=args.output_dir, goal_directed=args.goal_directed,
        )
        if args.benchmark == "guacamol"
        else evaluate_moses(samples, output_dir=args.output_dir, device=args.device)
    )
    result["samples_csv"] = str(args.samples_csv.resolve())
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
